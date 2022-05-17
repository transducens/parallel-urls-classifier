
import gc
import os
import sys
import time
import random
import logging
import argparse

import utils.utils as utils

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, WeightedRandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt

# Disable (less verbose) 3rd party logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

# Logging
logger = logging
logger_verbose = {"tokens": logging}

class URLsDataset(Dataset):
    def __init__(self, parallel_urls, non_parallel_urls, regression=False):
        #self.data = torch.stack(non_parallel_urls + parallel_urls).squeeze(1) # TODO problem here when creating a new tmp array -> big arrays will lead to run out of memory...
        self.data = non_parallel_urls + parallel_urls
        self.labels = np.zeros(len(self.data))
        #self.size_gb = self.data.element_size() * self.data.nelement() / 1000 / 1000 / 1000

        #self.labels[:len(non_parallel_urls)] = 0
        # Set to 1 the parallel URLs
        self.labels[len(non_parallel_urls):] = 1

        self.labels = torch.from_numpy(self.labels)
        self.labels = self.labels.type(torch.FloatTensor) if regression else self.labels.type(torch.LongTensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {"url_str": self.data[idx], "label": self.labels[idx]}

def get_confusion_matrix(outputs_argmax, labels, classes=2):
    tp, fp, fn, tn = np.zeros(classes), np.zeros(classes), np.zeros(classes), np.zeros(classes)

    for c in range(classes):
        # Multiclass confusion matrix
        # https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/
        tp[c] = torch.sum(torch.logical_and(labels == c, outputs_argmax == c))
        fp[c] = torch.sum(torch.logical_and(labels != c, outputs_argmax == c))
        fn[c] = torch.sum(torch.logical_and(labels == c, outputs_argmax != c))
        tn[c] = torch.sum(torch.logical_and(labels != c, outputs_argmax != c))

    return {"tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,}

def get_metrics(outputs_argmax, labels, current_batch_size, classes=2, idx=-1, log=False):
    acc = (torch.sum(outputs_argmax == labels) / current_batch_size).cpu().detach().numpy()

    no_values_per_class = np.zeros(classes)
    acc_per_class = np.zeros(classes)
    precision, recall, f1 = np.zeros(classes), np.zeros(classes), np.zeros(classes)
    macro_f1 = 0.0

    conf_mat = get_confusion_matrix(outputs_argmax, labels, classes=classes)
    tp, fp, fn, tn = conf_mat["tp"], conf_mat["fp"], conf_mat["fn"], conf_mat["tn"]

    for c in range(classes):
        no_values_per_class[c] = torch.sum(labels == c)

        # How many times have we classify correctly the target class taking into account all the data? -> we get how many percentage is from each class
        acc_per_class[c] = torch.sum(torch.logical_and(labels == c, outputs_argmax == c)) / current_batch_size

        # Metrics
        # http://rali.iro.umontreal.ca/rali/sites/default/files/publis/SokolovaLapalme-JIPM09.pdf
        # https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall
        precision[c] = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) != 0 else 1.0
        recall[c] = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) != 0 else 1.0
        f1[c] = 2 * ((precision[c] * recall[c]) / (precision[c] + recall[c])) if not np.isclose(precision[c] + recall[c], 0.0) else 0.0

    #assert outputs.shape[-1] == acc_per_class.shape[-1], f"Shape of outputs does not match the acc per class shape ({outputs.shape[-1]} vs {acc_per_class.shape[-1]})"
    assert np.isclose(np.sum(acc_per_class), acc), f"Acc and the sum of acc per classes should match ({acc} vs {np.sum(acc_per_class)})"

    macro_f1 = np.sum(f1) / f1.shape[0]

    if log:
        logger.debug("[train:batch#%d] Acc: %.2f %% (%.2f %% non-parallel and %.2f %% parallel)", idx + 1, acc * 100.0, acc_per_class[0] * 100.0, acc_per_class[1] * 100.0)
        logger.debug("[train:batch#%d] Acc per class (non-parallel->precision|recall|f1, parallel->precision|recall|f1): (%d -> %.2f %% | %.2f %% | %.2f %%, %d -> %.2f %% | %.2f %% | %.2f %%)",
                     idx + 1, no_values_per_class[0], precision[0] * 100.0, recall[0] * 100.0, f1[0] * 100.0, no_values_per_class[1], precision[1] * 100.0, recall[1] * 100.0, f1[1] * 100.0)
        logger.debug("[train:batch#%d] Macro F1: %.2f %%", idx + 1, macro_f1 * 100.0)

    return {"acc": acc,
            "acc_per_class": acc_per_class,
            "no_values_per_class": no_values_per_class,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_f1": macro_f1,}

@torch.no_grad()
def inference(model, tokenizer, criterion, dataloader, max_length_tokens, device, classes=2):
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_acc_per_class = np.zeros(2)
    total_acc_per_class_abs_precision = np.zeros(2)
    total_acc_per_class_abs_recall = np.zeros(2)
    total_acc_per_class_abs_f1 = np.zeros(2)
    total_macro_f1 = 0.0
    all_outputs = []
    all_labels = []

    for idx, batch in enumerate(dataloader):
        batch_urls_str = batch["url_str"]
        tokens = utils.encode(tokenizer, batch_urls_str, max_length_tokens)
        urls = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Inference
        outputs = model(urls, attention_mask).logits
        regression = outputs.cpu().detach().numpy().shape[1] == 1

        if regression:
            # Regression
            outputs = torch.sigmoid(outputs).squeeze()
            outputs_argmax = torch.round(outputs.cpu()).squeeze().numpy().astype(np.int64)
        else:
            # Binary classification
            outputs = F.softmax(outputs, dim=1)
            outputs_argmax = torch.argmax(outputs.cpu(), dim=1).numpy()

        # Results
        loss = criterion(outputs, labels).cpu().detach().numpy()
        labels = labels.cpu()

        if regression:
            labels = torch.round(labels).type(torch.LongTensor)

        total_loss += loss

        all_outputs.extend(outputs_argmax.tolist())
        all_labels.extend(labels.tolist())

    all_outputs = torch.tensor(all_outputs)
    all_labels = torch.tensor(all_labels)
    metrics = get_metrics(all_outputs, all_labels, len(all_labels), classes=classes)

    total_loss /= idx + 1
    total_acc += metrics["acc"]
    total_acc_per_class += metrics["acc_per_class"]
    total_acc_per_class_abs_precision += metrics["precision"]
    total_acc_per_class_abs_recall += metrics["recall"]
    total_acc_per_class_abs_f1 += metrics["f1"]
    total_macro_f1 += metrics["macro_f1"]

    return {"loss": total_loss,
            "acc": total_acc,
            "acc_per_class": total_acc_per_class,
            "precision": total_acc_per_class_abs_precision,
            "recall": total_acc_per_class_abs_recall,
            "f1": total_acc_per_class_abs_f1,
            "macro_f1": total_macro_f1,}

def plot_statistics(args, path=None, time_wait=5.0, freeze=False):
    plt_plot_common_params = {"marker": 'o', "markersize": 2,}
    plt_scatter_common_params = {"marker": 'o', "s": 2,}
    plt_legend_common_params = {"loc": "center left", "bbox_to_anchor": (1, 0.5), "fontsize": "x-small",}

    plt.clf()

    plt.subplot(3, 2, 1)
    plt.plot(list(map(lambda x: x * args["show_statistics_every_batches"], list(range(len(args["batch_loss"]))))), args["batch_loss"], label="Train loss", **plt_plot_common_params)
    plt.legend(**plt_legend_common_params)

    plt.subplot(3, 2, 5)
    plt.plot(list(map(lambda x: x * args["show_statistics_every_batches"], list(range(len(args["batch_acc"]))))), args["batch_acc"], label="Train acc", **plt_plot_common_params)
    plt.plot(list(map(lambda x: x * args["show_statistics_every_batches"], list(range(len(args["batch_acc_classes"][0]))))), args["batch_acc_classes"][0], label="Train F1: no p.", **plt_plot_common_params)
    plt.plot(list(map(lambda x: x * args["show_statistics_every_batches"], list(range(len(args["batch_acc_classes"][1]))))), args["batch_acc_classes"][1], label="Train F1: para.", **plt_plot_common_params)
    plt.plot(list(map(lambda x: x * args["show_statistics_every_batches"], list(range(len(args["batch_macro_f1"]))))), args["batch_macro_f1"], label="Train macro F1", **plt_plot_common_params)
    plt.legend(**plt_legend_common_params)

    plt.subplot(3, 2, 2)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_train_loss"], label="Train loss", **plt_plot_common_params)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_dev_loss"], label="Dev loss", **plt_plot_common_params)
    plt.legend(**plt_legend_common_params)

    plt.subplot(3, 2, 3)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_train_acc"], label="Train acc", **plt_plot_common_params)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_train_acc_classes"][0], label="Train F1: no p.", **plt_plot_common_params)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_train_acc_classes"][1], label="Train F1: para.", **plt_plot_common_params)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_train_macro_f1"], label="Train macro F1", **plt_plot_common_params)
    plt.legend(**plt_legend_common_params)

    plt.subplot(3, 2, 4)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_dev_acc"], label="Dev acc", **plt_plot_common_params)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_dev_acc_classes"][0], label="Dev F1: no p.", **plt_plot_common_params)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_dev_acc_classes"][1], label="Dev F1: para.", **plt_plot_common_params)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_dev_macro_f1"], label="Dev macro F1", **plt_plot_common_params)
    plt.legend(**plt_legend_common_params)

    plot_final = True if args["final_dev_acc"] else False

    plt.subplot(3, 2, 6)
    plt.scatter(0 if plot_final else None, args["final_dev_acc"] if plot_final else None, label="Dev acc", **plt_scatter_common_params)
    plt.scatter(0 if plot_final else None, args["final_test_acc"] if plot_final else None, label="Test acc", **plt_scatter_common_params)
    plt.scatter(1 if plot_final else None, args["final_dev_macro_f1"] if plot_final else None, label="Dev macro F1", **plt_scatter_common_params)
    plt.scatter(1 if plot_final else None, args["final_test_macro_f1"] if plot_final else None, label="Test macro F1", **plt_scatter_common_params)
    plt.legend(**plt_legend_common_params)

    #plt.tight_layout()
    plt.subplots_adjust(left=0.08,
                        bottom=0.07,
                        right=0.8,
                        top=0.95,
                        wspace=1.0,
                        hspace=0.4)

    if path:
        plt.savefig(path, dpi=1200)
    else:
        if freeze:
            plt.show()
        else:
            plt.pause(time_wait)

@torch.no_grad()
def interactive_inference(model, tokenizer, batch_size, max_length_tokens, device, inference_from_stdin=False, remove_authority=False,
                          parallel_likelihood=False, threshold=-np.inf, url_separator=' '):
    logger.info("Inference mode enabled")

    if not inference_from_stdin:
        logging.info("Insert 2 blank lines in order to end")

    logger_verbose["tokens"].debug("model_input\ttokens\ttokens2str\tunk_chars\tinitial_tokens_vs_detokenized\tinitial_tokens_vs_detokenized_len_1")

    model.eval()

    while True:
        if inference_from_stdin:
            try:
                target_urls, initial_urls = next(utils.tokenize_batch_from_fd(sys.stdin, tokenizer, batch_size, f=lambda u: utils.preprocess_url(u, remove_protocol_and_authority=remove_authority, separator=url_separator), return_urls=True))
            except StopIteration:
                break

            initial_src_urls = [u[0] for u in initial_urls]
            initial_trg_urls = [u[1] for u in initial_urls]
        else:
            initial_src_urls = [input("src url: ").strip()]
            initial_trg_urls = [input("trg url: ").strip()]

            if not initial_src_urls[0] and not initial_trg_urls[0]:
                break

            src_url = initial_src_urls[0]
            trg_url = initial_trg_urls[0]
            target_urls = next(utils.tokenize_batch_from_fd([f"{src_url}\t{trg_url}"], tokenizer, batch_size, f=lambda u: utils.preprocess_url(u, remove_protocol_and_authority=remove_authority, separator=url_separator)))

        # Tokens
        tokens = utils.encode(tokenizer, target_urls, max_length_tokens)
        urls = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        # Debug info
        ## Tokens
        urls_tokens = urls.cpu() * attention_mask.cpu()

        for ut in urls_tokens:
            url_tokens = ut[ut.nonzero()][:,0]
            original_str_from_tokens = tokenizer.decode(url_tokens) # Detokenize
            str_from_tokens = ' '.join([tokenizer.decode(t) for t in url_tokens]) # Detokenize adding a space between tokens
            ## Unk
            unk = torch.sum((url_tokens == tokenizer.unk_token_id).int()) # Unk tokens (this should happen just with very strange chars)
            sp_unk_vs_tokens_len = f"{len(original_str_from_tokens.split(url_separator))} vs {len(str_from_tokens.split(url_separator))}"
            sp_unk_vs_one_len_tokens = f"{sum(map(lambda u: 1 if len(u) == 1 else 0, original_str_from_tokens.split(url_separator)))} vs " \
                                       f"{sum(map(lambda u: 1 if len(u) == 1 else 0, str_from_tokens.split(url_separator)))}"

            logger_verbose["tokens"].debug("%s\t%s\t%s\t%d\t%s\t%s", original_str_from_tokens, str(url_tokens).replace('\n', ' '), str_from_tokens, unk, sp_unk_vs_tokens_len, sp_unk_vs_one_len_tokens)

        outputs = model(urls, attention_mask).logits
        regression = outputs.cpu().detach().numpy().shape[1] == 1

        if regression:
            # Regression
            outputs = torch.sigmoid(outputs).cpu().detach()
            outputs_argmax = torch.round(outputs).squeeze().numpy().astype(np.int64)
        else:
            # Binary classification
            outputs = F.softmax(outputs, dim=1).cpu().detach()
            outputs_argmax = torch.argmax(outputs, dim=1).numpy()

        if len(outputs_argmax.shape) == 0:
            outputs_argmax = np.array([outputs_argmax])

        assert outputs.numpy().shape[0] == len(initial_src_urls), f"Output samples does not match with the length of src URLs ({outputs.numpy().shape[0]} vs {initial_src_urls})"
        assert outputs.numpy().shape[0] == len(initial_trg_urls), f"Output samples does not match with the length of trg URLs ({outputs.numpy().shape[0]} vs {initial_trg_urls})"

        if parallel_likelihood:
            for data, initial_src_url, initial_trg_url in zip(outputs.numpy(), initial_src_urls, initial_trg_urls):
                likelihood = data[0] if regression else data[1] # parallel

                if likelihood >= threshold:
                    print(f"{likelihood:.4f}\t{initial_src_url}\t{initial_trg_url}")
        else:
            for argmax, initial_src_url, initial_trg_url in zip(outputs_argmax, initial_src_urls, initial_trg_urls):
                print(f"{'parallel' if argmax == 1 else 'non-parallel'}\t{initial_src_url}\t{initial_trg_url}")

def main(args):
    # https://discuss.pytorch.org/t/calculating-f1-score-over-batched-data/83348
    #logger.warning("Some metrics are calculated on each batch and averaged, so the values might not be fully correct (e.g. F1)")

    apply_inference = args.inference

    if not apply_inference:
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
    force_cpu = args.force_cpu
    device = torch.device("cuda:0" if use_cuda and not force_cpu else "cpu")
    pretrained_model = args.pretrained_model
    max_length_tokens = args.max_length_tokens
    model_input = utils.resolve_path(args.model_input)
    model_output = utils.resolve_path(args.model_output)
    seed = args.seed
    plot = args.plot
    plot_path = utils.resolve_path(args.plot_path)
    inference_from_stdin = args.inference_from_stdin
    parallel_likelihood = args.parallel_likelihood
    threshold = args.threshold
    imbalanced_strategy = args.imbalanced_strategy
    patience = args.patience
    do_not_load_best_model = args.do_not_load_best_model
    remove_authority = args.remove_authority
    add_symmetric_samples = args.add_symmetric_samples
    log_directory = args.log_directory
    regression = args.regression
    train_until_patience = args.train_until_patience
    url_separator = args.url_separator
    url_separator_new_token = args.url_separator_new_token
    warmup_steps = args.warmup_steps
    learning_rate = args.learning_rate
    stop_updating_lr_scheduler_proportion = args.stop_updating_lr_scheduler_proportion

    waiting_time = 20
    num_labels = 1 if regression else 2
    classes = 2

    # Disable parallelism since throws warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    if not utils.exists(log_directory, f=os.path.isdir):
        raise Exception(f"Provided log directory does not exist: '{log_directory}'")
    else:
        log_directory_files = os.listdir(utils.resolve_path(log_directory))

        if len(log_directory_files) != 0:
            logger.warning("Log directory contain %d files: waiting %d seconds before proceed", len(log_directory_files), waiting_time)

            time.sleep(waiting_time)

    # Loggers
    logger_verbose["tokens"] = logging.getLogger("parallel_urls_classifier.tokens")
    logger_verbose["tokens"].propagate = False # We don't want to see the messages multiple times
    logger_verbose["tokens"] = utils.set_up_logging_logger(logger_verbose["tokens"], level=logging.DEBUG if args.verbose else logging.INFO,
                                                           filename=f"{log_directory}/tokens", format="%(asctime)s\t%(levelname)s\t%(message)s")

    if not apply_inference and train_until_patience:
        logger.warning("Be aware that even training until patience is reached and not a fixed number of epochs, the selected number of epochs is relevant since it is a parameter which is used by the LR scheduler")

    if apply_inference and not model_input:
        logger.warning("Flag --model-input is recommended when --inference is provided: waiting %d seconds before proceed", waiting_time)

        time.sleep(waiting_time)

    logger.debug("Pretrained model architecture: %s", pretrained_model)

    if model_input and not utils.exists(model_input, f=os.path.isdir):
        raise Exception(f"Provided input model does not exist: '{model_input}'")
    if model_output:
        logger.info("Model will be stored: '%s'", model_output)

        if utils.exists(model_output, f=os.path.isdir):
            if args.overwrite_output_model:
                logger.warning("Provided output model does exist (file: '%s'): it will be updated: waiting %d seconds before proceed",
                                model_output, waiting_time)

                time.sleep(waiting_time)
            else:
                raise Exception(f"Provided output model does exist: '{model_output}'")

    if (do_not_load_best_model or not model_output) and not apply_inference:
        logger.warning("Final dev and test evaluation will not be carried out with the best model")

    if plot_path and not plot:
        raise Exception("--plot is mandatory if you set --plot-path")

    if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        logger.debug("Deterministic values enabled (not fully-guaranteed): seed %d", seed)
    else:
        logger.warning("Deterministic values disable (you set a negative seed)")

    if max_length_tokens > 512:
        logger.warning("HuggingFace models can handle a max. of 512 tokens at once and you set %d: changing value to 512")

        max_length_tokens = 512

    logger.info("Device: %s", device)

    if not apply_inference:
        logger.debug("Train URLs file (parallel, non-parallel): (%s, %s)", file_parallel_urls_train, file_non_parallel_urls_train)
        logger.debug("Dev URLs file (parallel, non-parallel): (%s, %s)", file_parallel_urls_dev, file_non_parallel_urls_dev)
        logger.debug("Test URLs file (parallel, non-parallel): (%s, %s)", file_parallel_urls_test, file_non_parallel_urls_test)

    model = AutoModelForSequenceClassification

    if model_input:
        logger.info("Loading model: '%s'", model_input)

        model = model.from_pretrained(model_input).to(device)
    else:
        model = model.from_pretrained(pretrained_model, num_labels=num_labels).to(device)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    fine_tuning = args.fine_tuning
    model_embeddings_size = model.base_model.embeddings.word_embeddings.weight.shape[0]

    if url_separator_new_token:
        # Add new special token (URL separator)
        num_added_toks = tokenizer.add_tokens([url_separator], special_tokens=True)

        logger.debug("New tokens added to tokenizer: %d", num_added_toks)

        if not model_input:
            model.resize_token_embeddings(len(tokenizer))
        elif model_embeddings_size + 1 == len(tokenizer):
            logger.warning("You've loaded a model which does not have the new token, so the results might be unexpected")

            model.resize_token_embeddings(len(tokenizer))

    model_embeddings_size = model.base_model.embeddings.word_embeddings.weight.shape[0]

    if model_embeddings_size != len(tokenizer):
        logger.error("Embedding layer size does not match with the tokenizer size: %d vs %d", model_embeddings_size, len(tokenizer))

    if apply_inference:
        interactive_inference(model, tokenizer, batch_size, max_length_tokens, device, inference_from_stdin=inference_from_stdin,
                              remove_authority=remove_authority, parallel_likelihood=parallel_likelihood, threshold=threshold,
                              url_separator=url_separator)

        # Stop execution
        return

    if regression:
        if imbalanced_strategy == "weighted-loss":
            logger.warning("Incompatible weight strategy ('%s'): it will not be applied", imbalanced_strategy)

    # Freeze layers, if needed
    for param in model.parameters():
        param.requires_grad = fine_tuning

    # Unfreeze classifier layer
    for param in model.classifier.parameters():
        param.requires_grad = True

    last_layer_output = utils.get_layer_from_model(model.base_model.encoder.layer[-1], name="output.dense.weight")

    logger.debug("Allocated memory before starting tokenization: %d", utils.get_current_allocated_memory_size())

    for fd, l in ((file_parallel_urls_train, parallel_urls_train), (file_non_parallel_urls_train, non_parallel_urls_train)):
        for idx, batch_urls in enumerate(utils.tokenize_batch_from_fd(fd, tokenizer, batch_size, f=lambda u: utils.preprocess_url(u, remove_protocol_and_authority=remove_authority, separator=url_separator), add_symmetric_samples=add_symmetric_samples)):
            l.extend(batch_urls)

    for fd, l in ((file_parallel_urls_dev, parallel_urls_dev), (file_non_parallel_urls_dev, non_parallel_urls_dev),
                  (file_parallel_urls_test, parallel_urls_test), (file_non_parallel_urls_test, non_parallel_urls_test)):
        for idx, batch_urls in enumerate(utils.tokenize_batch_from_fd(fd, tokenizer, batch_size, f=lambda u: utils.preprocess_url(u, remove_protocol_and_authority=remove_authority, separator=url_separator))):
            l.extend(batch_urls)

    logger.info("%d pairs of parallel URLs loaded (train)", len(parallel_urls_train))
    logger.info("%d pairs of non-parallel URLs loaded (train)", len(non_parallel_urls_train))
    logger.info("%d pairs of parallel URLs loaded (dev)", len(parallel_urls_dev))
    logger.info("%d pairs of non-parallel URLs loaded (dev)", len(non_parallel_urls_dev))
    logger.info("%d pairs of parallel URLs loaded (test)", len(parallel_urls_test))
    logger.info("%d pairs of non-parallel URLs loaded (test)", len(non_parallel_urls_test))

    min_train_samples = min(len(non_parallel_urls_train), len(parallel_urls_train))
    classes_count = np.array([len(non_parallel_urls_train), len(parallel_urls_train)]) # non-parallel URLs label is 0, and parallel URLs label is 1
    classes_weights = 1.0 / classes_count
    min_classes_weights = min_train_samples / classes_count

    if imbalanced_strategy == "none":
        # Is the data imbalanced? If so, warn about it

        for cw in min_classes_weights:
            if cw < 0.9:
                logger.warning("Your data seems to be imbalanced and you did not selected any imbalanced data strategy")
                break

    logger.debug("Classes weights: %s", str(classes_weights))

    classes_weights = torch.tensor(classes_weights, dtype=torch.float)

    no_workers = args.dataset_workers
    dataset_train = URLsDataset(parallel_urls_train, non_parallel_urls_train, regression=regression)

    if imbalanced_strategy == "over-sampling":
        target_list = []

        for t in dataset_train:
            l = t["label"]

            if regression:
                l = torch.round(l).type(torch.LongTensor)

            target_list.append(l)

        target_list = torch.tensor(target_list)
        classes_weights_all = classes_weights[target_list]

        # Over-sampling
        train_sampler = WeightedRandomSampler(
            weights=classes_weights_all,
            num_samples=len(classes_weights_all),
            replacement=True
        )
    else:
        train_sampler = RandomSampler(dataset_train)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler, num_workers=no_workers)
    dataset_dev = URLsDataset(parallel_urls_dev, non_parallel_urls_dev, regression=regression)
    dataloader_dev = DataLoader(dataset_dev, batch_size=batch_size, sampler=SequentialSampler(dataset_dev), num_workers=no_workers)
    dataset_test = URLsDataset(parallel_urls_test, non_parallel_urls_test, regression=regression)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, sampler=SequentialSampler(dataset_test), num_workers=no_workers)

    #logger.info("Train URLs: %.2f GB", dataset_train.size_gb)
    #logger.info("Dev URLs: %.2f GB", dataset_dev.size_gb)
    #logger.info("Test URLs: %.2f GB", dataset_test.size_gb)

    loss_weight = classes_weights if imbalanced_strategy == "weighted-loss" else None

    if regression:
        # Regression
        criterion = nn.MSELoss()
    else:
        # Binary classification
        criterion = nn.CrossEntropyLoss(weight=loss_weight)

    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=learning_rate, eps=1e-8)
    #optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
    #                  lr=2e-5,
    #                  eps=1e-8)

    if not warmup_steps:
        logger.warning("Warm-up steps set to %d since no value was set (set to 0 if that's the behaviour you were expecting)", len(dataloader_train))

        warmup_steps = len(dataloader_train)

    training_steps = len(dataloader_train) * epochs
    trainint_steps_freeze_lr = int((warmup_steps + training_steps) * stop_updating_lr_scheduler_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=training_steps)

    best_values_minimize = False
    best_values_maximize = True
    best_values_binary_func_comp = (lambda a, b: a > b) if best_values_minimize else (lambda a, b: a < b)

    assert best_values_minimize ^ best_values_maximize, "You can either minimize or maximize"

    logger.debug("Best values are being %s", "minimized" if best_values_minimize else "maximized")

    show_statistics_every_batches = 50
    final_loss = 0.0
    final_acc = 0.0
    final_acc_per_class = np.zeros(2)
    final_acc_per_class_abs = np.zeros(2)
    final_macro_f1 = 0.0
    best_dev = np.inf * (1 if best_values_minimize else -1)
    best_train = np.inf * (1 if best_values_minimize else -1)
    stop_training = False
    epoch = 0
    current_patience = 0
    current_training_step = 0

    # Statistics
    batch_loss = []
    batch_acc = []
    batch_acc_classes = {0: [], 1: []}
    batch_macro_f1 = []
    epoch_train_loss, epoch_dev_loss = [], []
    epoch_train_acc, epoch_dev_acc = [], []
    epoch_train_acc_classes, epoch_dev_acc_classes = {0: [], 1: []}, {0: [], 1: []}
    epoch_train_macro_f1, epoch_dev_macro_f1 = [], []

    while not stop_training:
        logger.info("Epoch %d", epoch + 1)

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_acc_per_class = np.zeros(2)
        epoch_acc_per_class_abs = np.zeros(2)
        epoch_macro_f1 = 0.0
        all_outputs = []
        all_labels = []

        model.train()

        for idx, batch in enumerate(dataloader_train):
            batch_urls_str = batch["url_str"]
            tokens = utils.encode(tokenizer, batch_urls_str, max_length_tokens)
            urls = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
            labels = batch["label"].to(device)
            current_batch_size = labels.reshape(-1).shape[0]

            #optimizer.zero_grad() # https://discuss.pytorch.org/t/model-zero-grad-or-optimizer-zero-grad/28426/6
            model.zero_grad()

            # Inference
            outputs = model(urls, attention_mask).logits

            if regression:
                # Regression
                outputs = torch.sigmoid(outputs).squeeze()
                outputs_argmax = torch.round(outputs.cpu()).squeeze().type(torch.int64)
            else:
                # Binary classification
                outputs = F.softmax(outputs, dim=1)
                outputs_argmax = torch.argmax(outputs.cpu(), dim=1)

            # Results
            loss = criterion(outputs, labels)
            loss_value = loss.cpu().detach().numpy()
            labels = labels.cpu()

            if regression:
                labels = torch.round(labels).type(torch.LongTensor)

            all_outputs.extend(outputs_argmax.tolist())
            all_labels.extend(labels.tolist())

            # Get metrics
            log = (idx + 1) % show_statistics_every_batches == 0
            metrics = get_metrics(outputs_argmax, labels, current_batch_size, classes=classes, idx=idx, log=log)

            if log:
                logger.debug("[train:batch#%d] Loss: %f", idx + 1, loss_value)

            epoch_loss += loss_value
            epoch_acc += metrics["acc"]
            epoch_acc_per_class += metrics["acc_per_class"]
            epoch_acc_per_class_abs += metrics["f1"]
            epoch_macro_f1 += metrics["macro_f1"]
            show_statistics = (epoch == 0 and idx == 0) or (idx + 1) % show_statistics_every_batches == 0

            if plot and show_statistics:
                utils.append_from_tuple((batch_loss, epoch_loss / (idx + 1)),
                                        (batch_acc, epoch_acc * 100.0 / (idx + 1)),
                                        (batch_acc_classes[0], epoch_acc_per_class_abs[0] * 100.0 / (idx + 1)),
                                        (batch_acc_classes[1], epoch_acc_per_class_abs[1] * 100.0 / (idx + 1)),
                                        (batch_macro_f1, epoch_macro_f1 * 100.0 / (idx + 1)))

                if epoch != 0 or idx != 0:
                    plot_args = {"show_statistics_every_batches": show_statistics_every_batches, "batch_loss": batch_loss,
                                 "batch_acc": batch_acc, "batch_acc_classes": batch_acc_classes, "batch_macro_f1": batch_macro_f1,
                                 "epoch": epoch, "epoch_train_loss": epoch_train_loss, "epoch_train_acc": epoch_train_acc,
                                 "epoch_train_acc_classes": epoch_train_acc_classes, "epoch_train_macro_f1": epoch_train_macro_f1,
                                 "epoch_dev_loss": epoch_dev_loss, "epoch_dev_acc": epoch_dev_acc, "epoch_dev_acc_classes": epoch_dev_acc_classes,
                                 "epoch_dev_macro_f1": epoch_dev_macro_f1, "final_dev_acc": None, "final_dev_macro_f1": None,
                                 "final_test_acc": None, "final_test_macro_f1": None,}

                    plot_statistics(plot_args, path=args.plot_path)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if show_statistics:
                # LRs statistics
                all_lrs = scheduler.get_last_lr()
                current_lr = all_lrs[0]
                len_lrs = len(all_lrs)

                if len_lrs != 1:
                    logger.debug("[batch#%d] First and last LRs: %s", idx + 1, f"{str(all_lrs[0:10])[:-1]} ... {str(all_lrs[10:])[1:]}")
                else:
                    logger.debug("[batch#%d] Current LR: %.8f", idx + 1, current_lr)

            if train_until_patience and current_training_step > trainint_steps_freeze_lr:
                # Do not update scheduler because we will reach lr = 0
                optimizer.step()

                if show_statistics:
                    logger.debug("[batch#%d] LR scheduler is not being updated", idx + 1)
            else:
                optimizer.step()
                scheduler.step()

            current_training_step += 1

        current_last_layer_output = utils.get_layer_from_model(model.base_model.encoder.layer[-1], name="output.dense.weight")
        layer_updated = (current_last_layer_output != last_layer_output).any().cpu().detach().numpy()

        logger.debug("Has the model layer been updated? %s", 'yes' if layer_updated else 'no')

        all_outputs = torch.tensor(all_outputs)
        all_labels = torch.tensor(all_labels)
        metrics = get_metrics(all_outputs, all_labels, len(all_labels), classes=classes)

        epoch_loss /= idx + 1
        epoch_acc = metrics["acc"]
        epoch_acc_per_class = metrics["acc_per_class"]
        epoch_acc_per_class_abs = metrics["f1"]
        epoch_macro_f1 = metrics["macro_f1"]
        final_loss += epoch_loss
        final_acc += epoch_acc
        final_acc_per_class += epoch_acc_per_class
        final_acc_per_class_abs += epoch_acc_per_class_abs
        final_macro_f1 += epoch_macro_f1

        logger.info("[train:epoch#%d] Avg. loss: %f", epoch + 1, epoch_loss)
        logger.info("[train:epoch#%d] Acc: %.2f %% (%.2f %% non-parallel and %.2f %% parallel)",
                    epoch + 1, epoch_acc * 100.0, epoch_acc_per_class[0] * 100.0, epoch_acc_per_class[1] * 100.0)
        logger.info("[train:epoch#%d] Acc per class (non-parallel:f1, parallel:f1): (%.2f %%, %.2f %%)",
                    epoch + 1, epoch_acc_per_class_abs[0] * 100.0, epoch_acc_per_class_abs[1] * 100.0)
        logger.info("[train:epoch#%d] Macro F1: %.2f %%", epoch + 1, epoch_macro_f1 * 100.0)

        dev_inference_metrics = inference(model, tokenizer, criterion, dataloader_dev, max_length_tokens, device, classes=classes)

        # Dev metrics
        dev_loss = dev_inference_metrics["loss"]
        dev_acc = dev_inference_metrics["acc"]
        dev_acc_per_class = dev_inference_metrics["acc_per_class"]
        dev_acc_per_class_abs_precision = dev_inference_metrics["precision"]
        dev_acc_per_class_abs_recall = dev_inference_metrics["recall"]
        dev_acc_per_class_abs_f1 = dev_inference_metrics["f1"]
        dev_macro_f1 = dev_inference_metrics["macro_f1"]

        logger.info("[dev:epoch#%d] Avg. loss: %f", epoch + 1, dev_loss)
        logger.info("[dev:epoch#%d] Acc: %.2f %% (%.2f %% non-parallel and %.2f %% parallel)",
                    epoch + 1, dev_acc * 100.0, dev_acc_per_class[0] * 100.0, dev_acc_per_class[1] * 100.0)
        logger.info("[dev:epoch#%d] Acc per class (non-parallel:precision|recall|f1, parallel:precision|recall|f1): (%.2f %% | %.2f %% | %.2f %%, %.2f %% | %.2f %% | %.2f %%)", epoch + 1,
                    dev_acc_per_class_abs_precision[0] * 100.0, dev_acc_per_class_abs_recall[0] * 100.0, dev_acc_per_class_abs_f1[0] * 100.0,
                    dev_acc_per_class_abs_precision[1] * 100.0, dev_acc_per_class_abs_recall[1] * 100.0, dev_acc_per_class_abs_f1[1] * 100.0)
        logger.info("[dev:epoch#%d] Macro F1: %.2f %%", epoch + 1, dev_macro_f1 * 100.0)

        # Get best dev and train result (check out best_values_minimize and best_values_maximize if you modify these values)
        dev_target = dev_macro_f1 # Might be acc, loss, ...
        train_target = epoch_macro_f1 # It should be the same metric that dev_target

        if best_values_binary_func_comp(best_dev, dev_target) or (best_dev == dev_target and best_values_binary_func_comp(best_train, train_target)):
            if best_dev == dev_target:
                logger.debug("Dev is equal but train has been improved from %s to %s: checkpoint", str(best_train), str(train_target))
            else:
                logger.debug("Dev has been improved from %s to %s: checkpoint", str(best_dev), str(dev_target))

            best_dev = dev_target

            if best_values_binary_func_comp(best_train, train_target):
                best_train = train_target

            # Store model
            if model_output:
                model.save_pretrained(model_output)

            current_patience = 0
        else:
            logger.debug("Dev has not been improved (best and current value): %s and %s", str(best_dev), str(dev_target))

            current_patience += 1

        if plot:
            utils.append_from_tuple((epoch_train_loss, epoch_loss),
                                    (epoch_train_acc, epoch_acc * 100.0),
                                    (epoch_train_acc_classes[0], epoch_acc_per_class_abs[0] * 100.0),
                                    (epoch_train_acc_classes[1], epoch_acc_per_class_abs[1] * 100.0),
                                    (epoch_train_macro_f1, epoch_macro_f1 * 100.0))
            utils.append_from_tuple((epoch_dev_loss, dev_loss),
                                    (epoch_dev_acc, dev_acc * 100.0),
                                    (epoch_dev_acc_classes[0], dev_acc_per_class_abs_f1[0] * 100.0),
                                    (epoch_dev_acc_classes[1], dev_acc_per_class_abs_f1[1] * 100.0),
                                    (epoch_dev_macro_f1, dev_macro_f1 * 100.0))

            plot_args = {"show_statistics_every_batches": show_statistics_every_batches, "batch_loss": batch_loss,
                         "batch_acc": batch_acc, "batch_acc_classes": batch_acc_classes, "batch_macro_f1": batch_macro_f1,
                         "epoch": epoch + 1, "epoch_train_loss": epoch_train_loss, "epoch_train_acc": epoch_train_acc,
                         "epoch_train_acc_classes": epoch_train_acc_classes, "epoch_train_macro_f1": epoch_train_macro_f1,
                         "epoch_dev_loss": epoch_dev_loss, "epoch_dev_acc": epoch_dev_acc, "epoch_dev_acc_classes": epoch_dev_acc_classes,
                         "epoch_dev_macro_f1": epoch_dev_macro_f1, "final_dev_acc": None, "final_dev_macro_f1": None,
                         "final_test_acc": None, "final_test_macro_f1": None,}

            plot_statistics(plot_args, path=args.plot_path)

        epoch += 1

        # Stop training?
        if patience > 0 and current_patience >= patience:
            # End of patience

            stop_training = True
        elif not train_until_patience:
            stop_training = epoch >= epochs

    final_loss /= epochs
    final_acc /= epochs
    final_acc_per_class /= epochs
    final_acc_per_class_abs /= epochs
    final_macro_f1 /= epochs

    logger.info("[train] Avg. loss: %f", final_loss)
    logger.info("[train] Avg. acc: %.2f %% (%.2f %% non-parallel and %.2f %% parallel)",
                final_acc * 100.0, final_acc_per_class[0] * 100.0, final_acc_per_class[1] * 100.0)
    logger.info("[train] Avg. acc per class (non-parallel:f1, parallel:f1): (%.2f %%, %.2f %%)",
                final_acc_per_class_abs[0] * 100.0, final_acc_per_class_abs[1] * 100.0)
    logger.info("[train] Avg. macro F1: %.2f %%", final_macro_f1 * 100.0)

    if do_not_load_best_model or not model_output:
        logger.warning("Using last model for dev and test evaluation")
    else:
        # Evaluate dev and test with best model

        logger.info("Loading best model: '%s' (best dev: %s)", model_output, str(best_dev))

        model = model.from_pretrained(model_output).to(device)

    dev_inference_metrics = inference(model, tokenizer, criterion, dataloader_dev, max_length_tokens, device, classes=classes)

    # Dev metrics
    dev_loss = dev_inference_metrics["loss"]
    dev_acc = dev_inference_metrics["acc"]
    dev_acc_per_class = dev_inference_metrics["acc_per_class"]
    dev_acc_per_class_abs_precision = dev_inference_metrics["precision"]
    dev_acc_per_class_abs_recall = dev_inference_metrics["recall"]
    dev_acc_per_class_abs_f1 = dev_inference_metrics["f1"]
    dev_macro_f1 = dev_inference_metrics["macro_f1"]

    logger.info("[dev] Avg. loss: %f", dev_loss)
    logger.info("[dev] Acc: %.2f %% (%.2f %% non-parallel and %.2f %% parallel)",
                dev_acc * 100.0, dev_acc_per_class[0] * 100.0, dev_acc_per_class[1] * 100.0)
    logger.info("[dev] Acc per class (non-parallel:precision|recall|f1, parallel:precision|recall|f1): (%.2f %% | %.2f %% | %.2f %%, %.2f %% | %.2f %% | %.2f %%)",
                dev_acc_per_class_abs_precision[0] * 100.0, dev_acc_per_class_abs_recall[0] * 100.0, dev_acc_per_class_abs_f1[0] * 100.0,
                dev_acc_per_class_abs_precision[1] * 100.0, dev_acc_per_class_abs_recall[1] * 100.0, dev_acc_per_class_abs_f1[1] * 100.0)
    logger.info("[dev] Macro F1: %.2f %%", dev_macro_f1 * 100.0)

    test_inference_metrics = inference(model, tokenizer, criterion, dataloader_test, max_length_tokens, device, classes=classes)

    # Test metrics
    test_loss = test_inference_metrics["loss"]
    test_acc = test_inference_metrics["acc"]
    test_acc_per_class = test_inference_metrics["acc_per_class"]
    test_acc_per_class_abs_precision = test_inference_metrics["precision"]
    test_acc_per_class_abs_recall = test_inference_metrics["recall"]
    test_acc_per_class_abs_f1 = test_inference_metrics["f1"]
    test_macro_f1 = test_inference_metrics["macro_f1"]

    logger.info("[test] Avg. loss: %f", test_loss)
    logger.info("[test] Acc: %.2f %% (%.2f %% non-parallel and %.2f %% parallel)",
                test_acc * 100.0, test_acc_per_class[0] * 100.0, test_acc_per_class[1] * 100.0)
    logger.info("[test] Acc per class (non-parallel:precision|recall|f1, parallel:precision|recall|f1): (%.2f %% | %.2f %% | %.2f %%, %.2f %% | %.2f %% | %.2f %%)",
                test_acc_per_class_abs_precision[0] * 100.0, test_acc_per_class_abs_recall[0] * 100.0, test_acc_per_class_abs_f1[0] * 100.0,
                test_acc_per_class_abs_precision[1] * 100.0, test_acc_per_class_abs_recall[1] * 100.0, test_acc_per_class_abs_f1[1] * 100.0)
    logger.info("[test] Macro F1: %.2f %%", test_macro_f1 * 100.0)

    if plot:
        plot_args = {"show_statistics_every_batches": show_statistics_every_batches, "batch_loss": batch_loss,
                     "batch_acc": batch_acc, "batch_acc_classes": batch_acc_classes, "batch_macro_f1": batch_macro_f1,
                     # '"epoch": epoch' and not '"epoch": epoch + 1' because we have not added new values
                     "epoch": epoch, "epoch_train_loss": epoch_train_loss, "epoch_train_acc": epoch_train_acc,
                     "epoch_train_acc_classes": epoch_train_acc_classes, "epoch_train_macro_f1": epoch_train_macro_f1,
                     "epoch_dev_loss": epoch_dev_loss, "epoch_dev_acc": epoch_dev_acc, "epoch_dev_acc_classes": epoch_dev_acc_classes,
                     "epoch_dev_macro_f1": epoch_dev_macro_f1, "final_dev_acc": dev_acc, "final_dev_macro_f1": dev_macro_f1,
                     "final_test_acc": test_acc, "final_test_macro_f1": test_macro_f1,}

        plot_statistics(plot_args, path=args.plot_path, freeze=True) # Let the user finish the execution if necessary

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Parallel URLs classifier")
    inference = "--inference" in sys.argv

    if not inference:
        parser.add_argument('parallel_urls_train_filename', type=argparse.FileType('rt'), help="Filename with parallel URLs (TSV format)")
        parser.add_argument('parallel_urls_dev_filename', type=argparse.FileType('rt'), help="Filename with parallel URLs (TSV format)")
        parser.add_argument('parallel_urls_test_filename', type=argparse.FileType('rt'), help="Filename with parallel URLs (TSV format)")
        parser.add_argument('non_parallel_urls_train_filename', type=argparse.FileType('rt'), help="Filename with non-parallel URLs (TSV format)")
        parser.add_argument('non_parallel_urls_dev_filename', type=argparse.FileType('rt'), help="Filename with non-parallel URLs (TSV format)")
        parser.add_argument('non_parallel_urls_test_filename', type=argparse.FileType('rt'), help="Filename with non-parallel URLs (TSV format)")

    parser.add_argument('--batch-size', type=int, default=16, help="Batch size")
    parser.add_argument('--epochs', type=int, default=3, help="Epochs")
    parser.add_argument('--fine-tuning', action="store_true", help="Apply fine-tuning")
    parser.add_argument('--dataset-workers', type=int, default=8, help="No. workers when loading the data in the dataset")
    parser.add_argument('--pretrained-model', default="xlm-roberta-base", help="Pretrained model")
    parser.add_argument('--max-length-tokens', type=int, default=256, help="Max. length for the generated tokens")
    parser.add_argument('--model-input', help="Model input path which will be loaded")
    parser.add_argument('--model-output', help="Model output path where the model will be stored")
    parser.add_argument('--inference', action="store_true", help="Do not train, just apply inference (flag --model-input is recommended)")
    parser.add_argument('--inference-from-stdin', action="store_true", help="Read inference from stdin")
    parser.add_argument('--parallel-likelihood', action="store_true", help="Print parallel likelihood instead of classification string (inference)")
    parser.add_argument('--threshold', type=float, default=-np.inf, help="Only print URLs which have a parallel likelihood greater than the provided threshold (inference)")
    parser.add_argument('--imbalanced-strategy', type=str, choices=["none", "over-sampling", "weighted-loss"], default="none", help="")
    parser.add_argument('--patience', type=int, default=0, help="Patience before stopping the training")
    parser.add_argument('--train-until-patience', action="store_true", help="Train until patience value is reached (--epochs will be ignored)")
    parser.add_argument('--do-not-load-best-model', action="store_true", help="Do not load best model for final dev and test evaluation (--model-output is necessary)")
    parser.add_argument('--overwrite-output-model', action="store_true", help="Overwrite output model if it exists (initial loading)")
    parser.add_argument('--remove-authority', action="store_true", help="Remove protocol and authority from provided URLs")
    parser.add_argument('--add-symmetric-samples', action="store_true", help="Add symmetric samples for training (if (src, trg) URL pair is provided, (trg, src) URL pair will be provided as well)")
    parser.add_argument('--force-cpu', action="store_true", help="Run on CPU (i.e. do not check if GPU is possible)")
    parser.add_argument('--log-directory', required=True, help="Directory where different log files will be stored")
    parser.add_argument('--regression', action="store_true", help="Apply regression instead of binary classification")
    parser.add_argument('--url-separator', default=' ', help="Separator to use when URLs are stringified")
    parser.add_argument('--url-separator-new-token', action="store_true", help="Add special token for URL separator")
    parser.add_argument('--warmup-steps', type=int, help="Warm-up steps")
    parser.add_argument('--learning-rate', type=float, default=2e-5, help="Warm-up steps")
    parser.add_argument('--stop-updating-lr-scheduler-proportion', type=float, default=1.0, help="Proportion that the LR scheduler will stop being updated, taking into account warming-up and training steps (e.g. if 0.5, the LR scheduler will stop being updated after half of training, related to epochs and batches, has been reached). If value >= 1.0, the LR will not stop being updated")

    parser.add_argument('--seed', type=int, default=71213, help="Seed in order to have deterministic results (not fully guaranteed). Set a negative number in order to disable this feature")
    parser.add_argument('--plot', action="store_true", help="Plot statistics (matplotlib pyplot) in real time")
    parser.add_argument('--plot-path', help="If set, the plot will be stored instead of displayed")

    parser.add_argument('-v', '--verbose', action="store_true", help="Verbose logging mode")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = initialization()

    logger = utils.set_up_logging_logger(logging.getLogger("parallel_urls_classifier"), level=logging.DEBUG if args.verbose else logging.INFO)

    logger.debug("Arguments processed: {}".format(str(args)))

    main(args)
