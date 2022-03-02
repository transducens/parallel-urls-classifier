
import gc
import os
import sys
import time
import random
import logging
import argparse
import urllib

import utils

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

class URLsDataset(Dataset):
    def __init__(self, parallel_urls, non_parallel_urls):
        #self.data = torch.stack(non_parallel_urls + parallel_urls).squeeze(1) # TODO problem here when creating a new tmp array -> big arrays will lead to run out of memory...
        self.data = non_parallel_urls + parallel_urls
        self.labels = np.zeros(len(self.data))
        #self.size_gb = self.data.element_size() * self.data.nelement() / 1000 / 1000 / 1000

        #self.labels[:len(non_parallel_urls)] = 0
        # Set to 1 the parallel URLs
        self.labels[len(non_parallel_urls):] = 1

        self.labels = torch.from_numpy(self.labels).type(torch.LongTensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {"bert_url_str": self.data[idx], "label": self.labels[idx]}

def inference(model, tokenizer, criterion, dataloader, max_length_tokens, device):
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_acc_per_class = np.zeros(2)
    total_acc_per_class_abs_precision = np.zeros(2)
    total_acc_per_class_abs_recall = np.zeros(2)
    total_acc_per_class_abs_f1 = np.zeros(2)

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch_urls_str = batch["bert_url_str"]
            tokens = utils.encode(tokenizer, batch_urls_str, max_length_tokens)
            urls = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
            #urls = batch["pair_urls"][:,0].to(device)
            #attention_mask = batch["pair_urls"][:,1].to(device)
            labels = batch["label"].to(device)
            current_batch_size = labels.reshape(-1).shape[0]

            #bert_output = apply_model(urls)
            #pooler_output = bert_output.pooler_output

            #outputs = model(pooler_output)
            outputs = model(urls, attention_mask).logits
            outputs = F.softmax(outputs, dim=1)
            outputs_argmax = torch.argmax(outputs.cpu(), dim=1)

            loss = criterion(outputs, labels).cpu().detach().numpy()
            labels = labels.cpu()
            acc = (torch.sum(outputs_argmax == labels) / current_batch_size).cpu().detach().numpy()
            acc_per_class = np.array([torch.sum(torch.logical_and(labels == 0, outputs_argmax == 0)),
                                      torch.sum(torch.logical_and(labels == 1, outputs_argmax == 1))]) / current_batch_size
            tp = np.array([torch.sum(torch.logical_and(labels == 0, outputs_argmax == 0)),
                           torch.sum(torch.logical_and(labels == 1, outputs_argmax == 1))])
            fp = np.array([torch.sum(torch.logical_and(labels != 0, outputs_argmax == 0)),
                           torch.sum(torch.logical_and(labels != 1, outputs_argmax == 1))])
            fn = np.array([torch.sum(torch.logical_and(labels == 0, outputs_argmax != 0)),
                           torch.sum(torch.logical_and(labels == 1, outputs_argmax != 1))])
            precision = np.array([tp[0] / (tp[0] + fp[0]) if (tp[0] + fp[0]) != 0 else 1.0 if tp[0] == 0 else 0.0,
                                  tp[1] / (tp[1] + fp[1]) if (tp[1] + fp[1]) != 0 else 1.0 if tp[1] == 0 else 0.0])
            recall = np.array([tp[0] / (tp[0] + fn[0]) if (tp[0] + fn[0]) != 0 else 1.0 if tp[0] == 0 else 0.0,
                               tp[1] / (tp[1] + fn[1]) if (tp[1] + fn[1]) != 0 else 1.0 if tp[1] == 0 else 0.0])
            f1 = np.array([2 * ((precision[0] * recall[0]) / (precision[0] + recall[0])) if (precision[0] + recall[0]) != 0 else 1.0 if (precision[0] * recall[0]) == 0 else 0.0,
                           2 * ((precision[1] * recall[1]) / (precision[1] + recall[1])) if (precision[1] + recall[1]) != 0 else 1.0 if (precision[1] * recall[1]) == 0 else 0.0])

            assert outputs.shape[-1] == acc_per_class.shape[-1], f"Shape of outputs does not match the acc per class shape ({outputs.shape[-1]} vs {acc_per_class.shape[-1]})"
            assert np.isclose(np.sum(acc_per_class), acc), f"Acc and the sum of acc per classes should match ({acc} vs {np.sum(acc_per_class)})"

            total_loss += loss
            total_acc += acc
            total_acc_per_class += acc_per_class
            total_acc_per_class_abs_precision += precision
            total_acc_per_class_abs_recall += recall
            total_acc_per_class_abs_f1 += f1

        total_loss /= idx + 1
        total_acc /= idx + 1
        total_acc_per_class /= idx + 1
        total_acc_per_class_abs_precision /= idx + 1
        total_acc_per_class_abs_recall /= idx + 1
        total_acc_per_class_abs_f1 /= idx + 1

    return total_loss, total_acc, total_acc_per_class, total_acc_per_class_abs_precision, \
           total_acc_per_class_abs_recall, total_acc_per_class_abs_f1

def plot_statistics(args, path=None, time_wait=5.0):
    plt.clf()

    plt.subplot(3, 2, 1)
    plt.plot(list(map(lambda x: x * args["show_statistics_every_batches"], list(range(len(args["batch_loss"]))))), args["batch_loss"], label="Train loss")
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(list(map(lambda x: x * args["show_statistics_every_batches"], list(range(len(args["batch_acc"]))))), args["batch_acc"], label="Train acc")
    plt.plot(list(map(lambda x: x * args["show_statistics_every_batches"], list(range(len(args["batch_acc_classes"][0]))))), args["batch_acc_classes"][0], label="Train F1 class 0")
    plt.plot(list(map(lambda x: x * args["show_statistics_every_batches"], list(range(len(args["batch_acc_classes"][1]))))), args["batch_acc_classes"][1], label="Train F1 class 1")
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(list(range(1, args["epochs"]))[:args["epoch"]], args["epoch_train_loss"], label="Train loss")
    plt.plot(list(range(1, args["epochs"]))[:args["epoch"]], args["epoch_dev_loss"], label="Dev loss")
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(list(range(1, args["epochs"]))[:args["epoch"]], args["epoch_train_acc"], label="Train acc")
    plt.plot(list(range(1, args["epochs"]))[:args["epoch"]], args["epoch_train_acc_classes"][0], label="Train F1 class 0")
    plt.plot(list(range(1, args["epochs"]))[:args["epoch"]], args["epoch_train_acc_classes"][1], label="Train F1 class 1")
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(list(range(1, args["epochs"]))[:args["epoch"]], args["epoch_dev_acc"], label="Dev acc")
    plt.plot(list(range(1, args["epochs"]))[:args["epoch"]], args["epoch_dev_acc_classes"][0], label="Dev F1 class 0")
    plt.plot(list(range(1, args["epochs"]))[:args["epoch"]], args["epoch_dev_acc_classes"][1], label="Dev F1 class 1")
    plt.legend()

    if path:
        plt.savefig(path, dpi=1200)
    else:
        plt.pause(time_wait)

def main(args):
    file_parallel_urls_train = args.parallel_urls_train_filename
    file_non_parallel_urls_train = args.non_parallel_urls_train_filename
    file_parallel_urls_dev = args.parallel_urls_dev_filename
    file_non_parallel_urls_dev = args.non_parallel_urls_dev_filename
    file_parallel_urls_test = args.parallel_urls_test_filename
    file_non_parallel_urls_test = args.non_parallel_urls_test_filename
    parallel_urls_train = []
    non_parallel_urls_train = []
    parallel_urls_dev = []
    non_parallel_urls_dev = []
    parallel_urls_test = []
    non_parallel_urls_test = []

    batch_size = args.batch_size
    epochs = args.epochs
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    pretrained_model = args.pretrained_model
    max_length_tokens = args.max_length_tokens
    model_input = utils.resolve_path(args.model_input)
    model_output = utils.resolve_path(args.model_output)
    seed = args.seed
    plot = args.plot
    plot_path = utils.resolve_path(args.plot_path)
    apply_inference = args.inference
    inference_from_stdin = args.inference_from_stdin
    waiting_time = 20

    if apply_inference and not model_input:
        logging.warning("Flag --model-input is recommended when --inference is provided: waiting %d seconds before proceed",
                        waiting_time)

        time.sleep(waiting_time)

    logging.debug("Pretrained model architecture: %s", pretrained_model)

    if model_input and not utils.exists(model_input):
        raise Exception(f"Provided input model does not exist: '{model_input}'")
    if model_output:
        logging.info("Model will be stored: '%s'", model_output)

        if utils.exists(model_output):
            logging.warning("Provided output model does exist (file: '%s'): waiting %d seconds before proceed",
                            model_input, waiting_time)

            time.sleep(waiting_time)

    if plot_path and not plot:
        raise Exception("--plot is mandatory if you set --plot-path")

    if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        logging.debug("Deterministic values enabled (not fully-guaranteed): seed %d", seed)
    else:
        logging.warning("Deterministic values disable (you set a negative seed)")

    if max_length_tokens > 512:
        logging.warning("BERT can handle a max. of 512 tokens at once and you set %d: changing value to 512")

        max_length_tokens = 512

    logging.info("Device: %s", device)
    logging.debug("Train URLs file (parallel, non-parallel): (%s, %s)", file_parallel_urls_train, file_non_parallel_urls_train)
    logging.debug("Dev URLs file (parallel, non-parallel): (%s, %s)", file_parallel_urls_dev, file_non_parallel_urls_dev)
    logging.debug("Test URLs file (parallel, non-parallel): (%s, %s)", file_parallel_urls_test, file_non_parallel_urls_test)

    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    fine_tuning = args.fine_tuning

    if model_input:
        logging.info("Loading model: '%s'", model_input)

        model.load_state_dict(torch.load(model_input))

    if apply_inference:
        logging.info("Inference mode enabled: insert 2 blank lines in order to end")

        while True:
            if inference_from_stdin:
                try:
                    src_and_trg_urls = next(sys.stdin).strip()
                except StopIteration:
                    break

                initial_src_url, initial_trg_url = src_and_trg_urls.split('\t')
            else:
                initial_src_url = input("src url: ").strip()
                initial_trg_url = input("trg url: ").strip()

                if not initial_src_url and not initial_trg_url:
                    break

            src_url = initial_src_url
            trg_url = initial_trg_url

            # Decode URL
            src_url = urllib.unquote(src_url)
            trg_url = urllib.unquote(trg_url)

            # Stringify URL
            src_url = utils.stringify_url(src_url)
            trg_url = utils.stringify_url(trg_url)

            # Input
            target_urls = f"{src_url} {tokenizer.sep_token} {trg_url}"
            # Tokens
            tokens = utils.encode(tokenizer, target_urls, max_length_tokens)
            urls = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)

            # Debug info
            ## Tokens
            urls_tokens = urls.cpu() * attention_mask.cpu()
            urls_tokens = urls_tokens[0][urls_tokens.nonzero()[:,1]] # Get unique batch and column 1 (pytorch format...)
            str_from_tokens = tokenizer.decode(urls_tokens) # Detokenize
            ## Unk
            unk = torch.sum((urls_tokens == tokenizer.unk_token_id).int()) # Unk tokens (this should happen just with very strange chars)
            sp_unk_vs_tokens_len = f"{len(target_urls.split(' '))} vs {len(str_from_tokens.split(' '))}"
            sp_unk_vs_one_len_tokens = f"{sum(map(lambda u: 1 if len(u) == 1 else 0, target_urls.split(' ')))} vs " \
                                       f"{sum(map(lambda u: 1 if len(u) == 1 else 0, str_from_tokens.split(' ')))}"

            logging.debug("Tokenization info (model input, from model input to tokens, from tokens to str): "
                          "(%s, %s, %s)", target_urls, str(urls_tokens).replace('\n', ' '), str_from_tokens)
            logging.debug("Unk. info (unk chars, initial tokens vs detokenized tokens, "
                          "len=1 -> initial tokens vs detokenized tokens): (%d, %s, %s)",
                          unk, sp_unk_vs_tokens_len, sp_unk_vs_one_len_tokens)

            outputs = model(urls, attention_mask).logits
            outputs = F.softmax(outputs, dim=1)
            outputs_argmax = torch.argmax(outputs.cpu(), dim=1).cpu().detach().numpy()[0]

            print(f"{'parallel' if outputs_argmax == 1 else 'non-parallel'}\t{initial_src_url}\t{initial_trg_url}")

        # Stop execution
        return

    # Freeze layers, if needed
    for param in model.parameters():
        param.requires_grad = fine_tuning

    # Unfreeze classifier layer
    for param in model.classifier.parameters():
        param.requires_grad = True

    bert_last_layer_output = utils.get_layer_from_model(model.base_model.encoder.layer[-1], name="output.dense.weight")

    logging.debug("Allocated memory before starting tokenization: %d", utils.get_current_allocated_memory_size())

    for fd, l in ((file_parallel_urls_train, parallel_urls_train), (file_non_parallel_urls_train, non_parallel_urls_train), 
                  (file_parallel_urls_dev, parallel_urls_dev), (file_non_parallel_urls_dev, non_parallel_urls_dev),
                  (file_parallel_urls_test, parallel_urls_test), (file_non_parallel_urls_test, non_parallel_urls_test)):
        for idx, batch_urls in enumerate(utils.tokenize_batch_from_fd(fd, tokenizer, batch_size * 10,
                                                                      f=lambda u: utils.stringify_url(urllib.unquote(u)))):
            l.extend(batch_urls)

    logging.info("%d pairs of parallel URLs loaded (train)", len(parallel_urls_train))
    logging.info("%d pairs of non-parallel URLs loaded (train)", len(non_parallel_urls_train))
    logging.info("%d pairs of parallel URLs loaded (dev)", len(parallel_urls_dev))
    logging.info("%d pairs of non-parallel URLs loaded (dev)", len(non_parallel_urls_dev))
    logging.info("%d pairs of parallel URLs loaded (test)", len(parallel_urls_test))
    logging.info("%d pairs of non-parallel URLs loaded (test)", len(non_parallel_urls_test))

    no_workers = args.dataset_workers
    dataset_train = URLsDataset(parallel_urls_train, non_parallel_urls_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=RandomSampler(dataset_train), num_workers=no_workers)
    dataset_dev = URLsDataset(parallel_urls_dev, non_parallel_urls_dev)
    dataloader_dev = DataLoader(dataset_dev, batch_size=batch_size, sampler=SequentialSampler(dataset_dev), num_workers=no_workers)
    dataset_test = URLsDataset(parallel_urls_test, non_parallel_urls_test)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, sampler=SequentialSampler(dataset_test), num_workers=no_workers)

    #logging.info("Train URLs: %.2f GB", dataset_train.size_gb)
    #logging.info("Dev URLs: %.2f GB", dataset_dev.size_gb)
    #logging.info("Test URLs: %.2f GB", dataset_test.size_gb)

    criterion = nn.CrossEntropyLoss().to(device)
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=2e-5,
                      eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train) * epochs)

    show_statistics_every_batches = 50
    final_loss = 0.0
    final_acc = 0.0
    final_acc_per_class = np.zeros(2)
    final_acc_per_class_abs = np.zeros(2)
    best_dev = -np.inf

    # Statistics
    batch_loss = []
    batch_acc = []
    batch_acc_classes = {0: [], 1: []}
    epoch_train_loss, epoch_dev_loss = [], []
    epoch_train_acc, epoch_dev_acc = [], []
    epoch_train_acc_classes, epoch_dev_acc_classes = {0: [], 1: []}, {0: [], 1: []}

    for epoch in range(epochs):
        logging.info("Epoch %d", epoch + 1)

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_acc_per_class = np.zeros(2)
        epoch_acc_per_class_abs = np.zeros(2)

        model.train()

        for idx, batch in enumerate(dataloader_train):
            batch_urls_str = batch["bert_url_str"]
            tokens = utils.encode(tokenizer, batch_urls_str, max_length_tokens)
            urls = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
            labels = batch["label"].to(device)
            current_batch_size = labels.reshape(-1).shape[0]

            #optimizer.zero_grad() # https://discuss.pytorch.org/t/model-zero-grad-or-optimizer-zero-grad/28426/6
            model.zero_grad()

            #bert_output = apply_model(urls)
            #last_hidden_state = bert_output.last_hidden_state
            #pooler_output = bert_output.pooler_output

            #outputs = model(pooler_output)
            outputs = model(urls, attention_mask).logits
            outputs = F.softmax(outputs, dim=1)
            outputs_argmax = torch.argmax(outputs.cpu(), dim=1)

            loss = criterion(outputs, labels)
            loss_value = loss.cpu().detach().numpy()
            labels = labels.cpu()
            acc = (torch.sum(outputs_argmax == labels) / current_batch_size).cpu().detach().numpy()
            # How many times have we classify correctly the target class taking into account all the data? -> we get how many percentage is from each class
            acc_per_class = np.array([torch.sum(torch.logical_and(labels == 0, outputs_argmax == 0)),
                                      torch.sum(torch.logical_and(labels == 1, outputs_argmax == 1))]) / current_batch_size
            # How many times have we classify correctly the target class taking into account only the data of the specific class?
            tp = np.array([torch.sum(torch.logical_and(labels == 0, outputs_argmax == 0)),
                           torch.sum(torch.logical_and(labels == 1, outputs_argmax == 1))])
            fp = np.array([torch.sum(torch.logical_and(labels != 0, outputs_argmax == 0)),
                           torch.sum(torch.logical_and(labels != 1, outputs_argmax == 1))])
            fn = np.array([torch.sum(torch.logical_and(labels == 0, outputs_argmax != 0)),
                           torch.sum(torch.logical_and(labels == 1, outputs_argmax != 1))])
            precision = np.array([tp[0] / (tp[0] + fp[0]) if (tp[0] + fp[0]) != 0 else 1.0 if tp[0] == 0 else 0.0,
                                  tp[1] / (tp[1] + fp[1]) if (tp[1] + fp[1]) != 0 else 1.0 if tp[1] == 0 else 0.0])
            recall = np.array([tp[0] / (tp[0] + fn[0]) if (tp[0] + fn[0]) != 0 else 1.0 if tp[0] == 0 else 0.0,
                               tp[1] / (tp[1] + fn[1]) if (tp[1] + fn[1]) != 0 else 1.0 if tp[1] == 0 else 0.0])
            f1 = np.array([2 * ((precision[0] * recall[0]) / (precision[0] + recall[0])) if (precision[0] + recall[0]) != 0 else 1.0 if (precision[0] * recall[0]) == 0 else 0.0,
                           2 * ((precision[1] * recall[1]) / (precision[1] + recall[1])) if (precision[1] + recall[1]) != 0 else 1.0 if (precision[1] * recall[1]) == 0 else 0.0])

            assert outputs.shape[-1] == acc_per_class.shape[-1], f"Shape of outputs does not match the acc per class shape ({outputs.shape[-1]} vs {acc_per_class.shape[-1]})"
            assert np.isclose(np.sum(acc_per_class), acc), f"Acc and the sum of acc per classes should match ({acc} vs {np.sum(acc_per_class)})"

            if (idx + 1) % show_statistics_every_batches == 0:
                logging.debug("[train:batch#%d] Loss: %f", idx + 1, loss_value)
                logging.debug("[train:batch#%d] Acc: %.2f %% (%.2f %% non-parallel and %.2f %% parallel)", idx + 1, acc * 100.0, acc_per_class[0] * 100.0, acc_per_class[1] * 100.0)
                logging.debug("[train:batch#%d] Acc per class (non-parallel:precision|recall|f1, parallel:precision|recall|f1): (%.2f %% | %.2f %% | %.2f %%, %.2f %% | %.2f %% | %.2f %%)",
                              idx + 1, precision[0] * 100.0, recall[0] * 100.0, f1[0] * 100.0, precision[1] * 100.0, recall[1] * 100.0, f1[1] * 100.0)

            epoch_loss += loss_value
            epoch_acc += acc
            epoch_acc_per_class += acc_per_class
            epoch_acc_per_class_abs += f1

            if epoch == 0 and idx == 0:
                utils.append_from_tuple((batch_loss, epoch_loss),
                                        (batch_acc, epoch_acc * 100.0),
                                        (batch_acc_classes[0], epoch_acc_per_class_abs[0] * 100.0),
                                        (batch_acc_classes[1], epoch_acc_per_class_abs[1] * 100.0))
            elif plot and (idx + 1) % show_statistics_every_batches == 0:
                utils.append_from_tuple((batch_loss, epoch_loss / (idx + 1)),
                                        (batch_acc, epoch_acc * 100.0 / (idx + 1)),
                                        (batch_acc_classes[0], epoch_acc_per_class_abs[0] * 100.0 / (idx + 1)),
                                        (batch_acc_classes[1], epoch_acc_per_class_abs[1] * 100.0 / (idx + 1)))
                
                plot_args = {"show_statistics_every_batches": show_statistics_every_batches, "batch_loss": batch_loss,
                             "batch_acc": batch_acc, "batch_acc_classes": batch_acc_classes, "epochs": epochs,
                             "epoch": epoch, "epoch_train_loss": epoch_train_loss, "epoch_train_acc": epoch_train_acc,
                             "epoch_train_acc_classes": epoch_train_acc_classes, "epoch_dev_loss": epoch_dev_loss,
                             "epoch_dev_acc": epoch_dev_acc, "epoch_dev_acc_classes": epoch_dev_acc_classes}

                plot_statistics(plot_args, path=args.plot_path)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        current_last_layer_output = utils.get_layer_from_model(model.base_model.encoder.layer[-1], name="output.dense.weight")
        layer_updated = (current_last_layer_output != bert_last_layer_output).any().cpu().detach().numpy()

        logging.debug("Has the model layer been updated? %s", 'yes' if layer_updated else 'no')

        epoch_loss /= idx + 1
        epoch_acc /= idx + 1
        epoch_acc_per_class /= idx + 1
        epoch_acc_per_class_abs /= idx + 1
        final_loss += epoch_loss
        final_acc += epoch_acc
        final_acc_per_class += epoch_acc_per_class
        final_acc_per_class_abs += epoch_acc_per_class_abs

        logging.info("[train:epoch#%d] Loss: %f", epoch + 1, epoch_loss)
        logging.info("[train:epoch#%d] Acc: %.2f %% (%.2f %% non-parallel and %.2f %% parallel)",
                     epoch + 1, epoch_acc * 100.0, epoch_acc_per_class[0] * 100.0, epoch_acc_per_class[1] * 100.0)
        logging.info("[train:epoch#%d] Acc per class (non-parallel:f1, parallel:f1): (%.2f %%, %.2f %%)",
                     epoch + 1, epoch_acc_per_class_abs[0] * 100.0, epoch_acc_per_class_abs[1] * 100.0)

        dev_loss, dev_acc, dev_acc_per_class, dev_acc_per_class_abs_precision, dev_acc_per_class_abs_recall, dev_acc_per_class_abs_f1 = \
            inference(model, tokenizer, criterion, dataloader_dev, max_length_tokens, device)

        logging.info("[dev:epoch#%d] Loss: %f", epoch + 1, dev_loss)
        logging.info("[dev:epoch#%d] Acc: %.2f %% (%.2f %% non-parallel and %.2f %% parallel)",
                     epoch + 1, dev_acc * 100.0, dev_acc_per_class[0] * 100.0, dev_acc_per_class[1] * 100.0)
        logging.info("[dev:epoch#%d] Acc per class (non-parallel:precision|recall|f1, parallel:precision|recall|f1): (%.2f %% | %.2f %% | %.2f %%, %.2f %% | %.2f %% | %.2f %%)", epoch + 1,
                     dev_acc_per_class_abs_precision[0] * 100.0, dev_acc_per_class_abs_recall[0] * 100.0, dev_acc_per_class_abs_f1[0] * 100.0,
                     dev_acc_per_class_abs_precision[1] * 100.0, dev_acc_per_class_abs_recall[1] * 100.0, dev_acc_per_class_abs_f1[1] * 100.0)

        # Get best dev result
        dev_target = dev_acc # Might be acc, loss, ...

        if best_dev < dev_target:
            logging.debug("Dev has been improved: from %s to %s", str(best_dev), str(dev_target))

            best_dev = dev_target

            # Store model
            if model_output:
                torch.save(model.state_dict(), model_output)

        if plot:
            utils.append_from_tuple((epoch_train_loss, epoch_loss),
                                    (epoch_train_acc, epoch_acc * 100.0),
                                    (epoch_train_acc_classes[0], epoch_acc_per_class_abs[0] * 100.0),
                                    (epoch_train_acc_classes[1], epoch_acc_per_class_abs[1] * 100.0))
            utils.append_from_tuple((epoch_dev_loss, dev_loss),
                                    (epoch_dev_acc, dev_acc * 100.0),
                                    (epoch_dev_acc_classes[0], dev_acc_per_class_abs_f1[0] * 100.0),
                                    (epoch_dev_acc_classes[1], dev_acc_per_class_abs_f1[1] * 100.0))

            plot_args = {"show_statistics_every_batches": show_statistics_every_batches, "batch_loss": batch_loss,
                         "batch_acc": batch_acc, "batch_acc_classes": batch_acc_classes, "epochs": epochs + 1,
                         "epoch": epoch + 1, "epoch_train_loss": epoch_train_loss, "epoch_train_acc": epoch_train_acc,
                         "epoch_train_acc_classes": epoch_train_acc_classes, "epoch_dev_loss": epoch_dev_loss,
                         "epoch_dev_acc": epoch_dev_acc, "epoch_dev_acc_classes": epoch_dev_acc_classes}

            plot_statistics(plot_args, path=args.plot_path)

    final_loss /= epochs
    final_acc /= epochs
    final_acc_per_class /= epochs
    final_acc_per_class_abs /= epochs

    logging.info("[train] Loss: %f", final_loss)
    logging.info("[train] Acc: %.2f %% (%.2f %% non-parallel and %.2f %% parallel)",
                 final_acc * 100.0, final_acc_per_class[0] * 100.0, final_acc_per_class[1] * 100.0)
    logging.info("[train] Acc per class (non-parallel:f1, parallel:f1): (%.2f %%, %.2f %%)",
                 final_acc_per_class_abs[0] * 100.0, final_acc_per_class_abs[1] * 100.0)

    dev_loss, dev_acc, dev_acc_per_class, dev_acc_per_class_abs_precision, dev_acc_per_class_abs_recall, dev_acc_per_class_abs_f1 = \
        inference(model, tokenizer, criterion, dataloader_dev, max_length_tokens, device)

    logging.info("[dev] Loss: %f", dev_loss)
    logging.info("[dev] Acc: %.2f %% (%.2f %% non-parallel and %.2f %% parallel)",
                 dev_acc * 100.0, dev_acc_per_class[0] * 100.0, dev_acc_per_class[1] * 100.0)
    logging.info("[dev] Acc per class (non-parallel:precision|recall|f1, parallel:precision|recall|f1): (%.2f %% | %.2f %% | %.2f %%, %.2f %% | %.2f %% | %.2f %%)",
                 dev_acc_per_class_abs_precision[0] * 100.0, dev_acc_per_class_abs_recall[0] * 100.0, dev_acc_per_class_abs_f1[0] * 100.0,
                 dev_acc_per_class_abs_precision[1] * 100.0, dev_acc_per_class_abs_recall[1] * 100.0, dev_acc_per_class_abs_f1[1] * 100.0)

    test_loss, test_acc, test_acc_per_class, test_acc_per_class_abs_precision, test_acc_per_class_abs_recall, test_acc_per_class_abs_f1 = \
        inference(model, tokenizer, criterion, dataloader_test, max_length_tokens, device)

    logging.info("[test] Loss: %f", test_loss)
    logging.info("[test] Acc: %.2f %% (%.2f %% non-parallel and %.2f %% parallel)",
                 test_acc * 100.0, test_acc_per_class[0] * 100.0, test_acc_per_class[1] * 100.0)
    logging.info("[test] Acc per class (non-parallel:precision|recall|f1, parallel:precision|recall|f1): (%.2f %% | %.2f %% | %.2f %%, %.2f %% | %.2f %% | %.2f %%)",
                 test_acc_per_class_abs_precision[0] * 100.0, test_acc_per_class_abs_recall[0] * 100.0, test_acc_per_class_abs_f1[0] * 100.0,
                 test_acc_per_class_abs_precision[1] * 100.0, test_acc_per_class_abs_recall[1] * 100.0, test_acc_per_class_abs_f1[1] * 100.0)

    if not args.plot_path:
        # Let the user finish the execution
        plt.show()

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="URLs classifier")

    parser.add_argument('parallel_urls_train_filename', type=argparse.FileType('rt'), help="Filename with parallel URLs (TSV format)")
    parser.add_argument('parallel_urls_dev_filename', type=argparse.FileType('rt'), help="Filename with parallel URLs (TSV format)")
    parser.add_argument('parallel_urls_test_filename', type=argparse.FileType('rt'), help="Filename with parallel URLs (TSV format)")
    parser.add_argument('non_parallel_urls_train_filename', type=argparse.FileType('rt'), help="Filename with non-parallel URLs (TSV format)")
    parser.add_argument('non_parallel_urls_dev_filename', type=argparse.FileType('rt'), help="Filename with non-parallel URLs (TSV format)")
    parser.add_argument('non_parallel_urls_test_filename', type=argparse.FileType('rt'), help="Filename with non-parallel URLs (TSV format)")

    parser.add_argument('--batch-size', type=int, default=16, help="Batch size")
    parser.add_argument('--epochs', type=int, default=3, help="Epochs")
    parser.add_argument('--fine-tuning', action='store_true', help="Apply fine-tuning")
    parser.add_argument('--dataset-workers', type=int, default=8, help="No. workers when loading the data in the dataset")
    parser.add_argument('--pretrained-model', default="xlm-roberta-base", help="Pretrained model")
    parser.add_argument('--max-length-tokens', type=int, default=256, help="Max. length for the generated tokens")
    parser.add_argument('--model-input', help="Model input path which will be loaded")
    parser.add_argument('--model-output', help="Model output path where the model will be stored")
    parser.add_argument('--inference', action="store_true", help="Do not train, just apply inference (flag --model-input is recommended)")
    parser.add_argument('--inference-from-stdin', action="store_true", help="Read inference from stdin")

    parser.add_argument('--seed', type=int, default=71213, help="Seed in order to have deterministic results (not fully guaranteed). Set a negative number in order to disable this feature")
    parser.add_argument('--plot', action='store_true', help="Plot statistics (matplotlib pyplot) in real time")
    parser.add_argument('--plot-path', help="If set, the plot will be stored instead of displayed")

    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose logging mode")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = initialization()

    utils.set_up_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    logging.debug("Arguments processed: {}".format(str(args)))

    main(args)
