
import sys

import utils.utils as utils
from metrics import (
    get_metrics,
)

import numpy as np
import torch
import torch.nn.functional as F

@torch.no_grad()
def inference(model, tokenizer, criterion, dataloader, max_length_tokens, device, amp_context_manager, logger, classes=2):
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
        with amp_context_manager:
            outputs = model(urls, attention_mask).logits

            regression = outputs.cpu().detach().numpy().shape[1] == 1

            if regression:
                # Regression
                outputs = torch.sigmoid(outputs).squeeze()
                outputs_argmax = torch.round(outputs).squeeze().cpu().numpy().astype(np.int64) # Workaround for https://github.com/pytorch/pytorch/issues/54774
            else:
                # Binary classification
                outputs = F.softmax(outputs, dim=1)
                outputs_argmax = torch.argmax(outputs.cpu(), dim=1).numpy()

            loss = criterion(outputs, labels).cpu().detach().numpy()

        labels = labels.cpu()

        if regression:
            labels = torch.round(labels).type(torch.LongTensor)

        total_loss += loss

        all_outputs.extend(outputs_argmax.tolist())
        all_labels.extend(labels.tolist())

    all_outputs = torch.tensor(all_outputs)
    all_labels = torch.tensor(all_labels)
    metrics = get_metrics(all_outputs, all_labels, len(all_labels), logger, classes=classes)

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

@torch.no_grad()
def interactive_inference(model, tokenizer, batch_size, max_length_tokens, device, amp_context_manager, logger, logger_verbose,
                          inference_from_stdin=False, remove_authority=False, remove_positional_data_from_resource=False,
                          parallel_likelihood=False, threshold=-np.inf, url_separator=' '):
    logger.info("Inference mode enabled")

    if not inference_from_stdin:
        logger.info("Insert 2 blank lines in order to end")

    logger_verbose["tokens"].debug("model_input\ttokens\ttokens2str\tunk_chars\tinitial_tokens_vs_detokenized\tinitial_tokens_vs_detokenized_len_1")

    model.eval()

    while True:
        if inference_from_stdin:
            try:
                target_urls, initial_urls = next(utils.tokenize_batch_from_fd(sys.stdin, tokenizer, batch_size, f=lambda u: utils.preprocess_url(u, remove_protocol_and_authority=remove_authority, remove_positional_data=remove_positional_data_from_resource, separator=url_separator), return_urls=True))
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
            target_urls = next(utils.tokenize_batch_from_fd([f"{src_url}\t{trg_url}"], tokenizer, batch_size, f=lambda u: utils.preprocess_url(u, remove_protocol_and_authority=remove_authority, remove_positional_data=remove_positional_data_from_resource, separator=url_separator)))

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

        with amp_context_manager:
            outputs = model(urls, attention_mask).logits

        regression = outputs.cpu().detach().numpy().shape[1] == 1

        if regression:
            # Regression
            outputs = torch.sigmoid(outputs).detach()
            outputs_argmax = torch.round(outputs).squeeze().cpu().numpy().astype(np.int64)
        else:
            # Binary classification
            outputs = F.softmax(outputs, dim=1).detach()
            outputs_argmax = torch.argmax(outputs, dim=1).cpu().numpy()

        outputs = outputs.cpu()

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
