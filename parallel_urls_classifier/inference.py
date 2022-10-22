
import sys

import parallel_urls_classifier.utils.utils as utils
from parallel_urls_classifier.metrics import (
    get_metrics,
)
import parallel_urls_classifier.preprocess as preprocess

import numpy as np
import torch
import torch.nn.functional as F

def inference_with_heads(model, all_heads, criteria, urls, attention_mask, labels, regression, amp_context_manager):
    if len(all_heads) != len(criteria):
        raise Exception(f"Different length in the provided heads and criteria: {len(all_heads)} vs {len(criteria)}")

    results = []

    with amp_context_manager:
        model_outputs = model(urls, attention_mask)

        for idx in range(len(all_heads)):
            # TODO fix for fit different heads
            head = all_heads[idx]
            criterion = criteria[idx]
            head_wrapper_name = head.head_variable_name
            getattr(head, head_wrapper_name).set_tensor_for_returning(model_outputs) # Set the output of the model -> don't execute again the model

            outputs = head(None).logits # Get head result

            if regression:
                # Regression
                outputs = torch.sigmoid(outputs).squeeze(1)
                outputs_argmax = torch.round(outputs).type(torch.int64).cpu() # Workaround for https://github.com/pytorch/pytorch/issues/54774
            else:
                # Binary classification
                outputs_argmax = torch.argmax(F.softmax(outputs, dim=1).cpu(), dim=1)

            loss = criterion(outputs, labels)

            results.append((outputs, outputs_argmax, loss))

    return results

@torch.no_grad()
def inference(model, all_heads, tokenizer, criterion, dataloader, max_length_tokens, device, regression,
              amp_context_manager, logger, classes=2):
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
        results = inference_with_heads(model, all_heads, [criterion], urls, attention_mask, labels, regression, amp_context_manager)

        # TODO fix
        outputs, outputs_argmax, loss = results[0]

        loss = loss.cpu()
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
                          parallel_likelihood=False, threshold=-np.inf, url_separator=' ', lower=True):
    logger.info("Inference mode enabled")

    if not inference_from_stdin:
        logger.info("Insert 2 blank lines in order to end")

    logger_verbose["tokens"].debug("preprocessed_urls\tmodel_input\ttokens\ttokens2str\tunk_chars\t" \
                                   "initial_tokens_vs_detokenized\tinitial_tokens_vs_detokenized_len_1")

    model.eval()

    pad_token = utils.encode(tokenizer, tokenizer.pad_token, 1, False).input_ids.squeeze()

    while True:
        if inference_from_stdin:
            try:
                target_urls, initial_urls = \
                    next(utils.tokenize_batch_from_fd(sys.stdin, tokenizer, batch_size,
                            f=lambda u: preprocess.preprocess_url(u, remove_protocol_and_authority=remove_authority,
                                                                  remove_positional_data=remove_positional_data_from_resource,
                                                                  separator=url_separator, lower=lower),
                            return_urls=True))

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
            target_urls = next(utils.tokenize_batch_from_fd([f"{src_url}\t{trg_url}"],
                               tokenizer, batch_size,
                               f=lambda u: preprocess.preprocess_url(u, remove_protocol_and_authority=remove_authority,
                                                                     remove_positional_data=remove_positional_data_from_resource,
                                                                     separator=url_separator, lower=lower)))

        # Tokens
        tokens = utils.encode(tokenizer, target_urls, max_length_tokens)
        urls = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        # Debug info
        ## Tokens
        urls_tokens = urls.cpu() * attention_mask.cpu() # PAD tokens -> 0

        for idx, ut in enumerate(urls_tokens):
            url_tokens = ut[ut != pad_token]
            original_str_from_tokens = tokenizer.decode(url_tokens) # Detokenize
            str_from_tokens = '<tok_sep>'.join([tokenizer.decode(t) for t in url_tokens]) # Detokenize adding a mark between tokens
            ## Unk
            unk = torch.sum((url_tokens == tokenizer.unk_token_id).int()) # Unk tokens (this should happen just with very strange chars)
            sp_unk_vs_tokens_len = f"{len(original_str_from_tokens.split(url_separator))} vs " \
                                   f"{len(str_from_tokens.split(url_separator))}"
            sp_unk_vs_one_len_tokens = f"{sum(map(lambda u: 1 if len(u) == 1 else 0, original_str_from_tokens.split(url_separator)))} vs " \
                                       f"{sum(map(lambda u: 1 if len(u) == 1 else 0, str_from_tokens.split(url_separator)))}"

            logger_verbose["tokens"].debug("%s\t%s\t%s\t%s\t%d\t%s\t%s", target_urls[idx], original_str_from_tokens, str(url_tokens).replace('\n', ' '), str_from_tokens, unk, sp_unk_vs_tokens_len, sp_unk_vs_one_len_tokens)

        with amp_context_manager:
            outputs = model(urls, attention_mask).logits

        regression = outputs.cpu().detach().numpy().shape[1] == 1

        if regression:
            # Regression
            outputs = torch.sigmoid(outputs).detach()
            outputs_argmax = torch.round(outputs).squeeze(1).cpu().numpy().astype(np.int64)
        else:
            # Binary classification
            outputs = outputs.detach()
            outputs_argmax = torch.argmax(F.softmax(outputs, dim=1), dim=1).cpu().numpy()

        outputs = outputs.cpu()

        # TODO TEST_BEFORE if we use outputs_argmax.squeeze(1) instead of outputs_argmax.squeeze(), the following condition should be safe to be removed
        if len(outputs_argmax.shape) == 0:
            outputs_argmax = np.array([outputs_argmax])

        assert outputs.numpy().shape[0] == len(initial_src_urls), "Output samples does not match with the length of src URLs " \
                                                                  f"({outputs.numpy().shape[0]} vs {len(initial_src_urls)})"
        assert outputs.numpy().shape[0] == len(initial_trg_urls), "Output samples does not match with the length of trg URLs " \
                                                                  f"({outputs.numpy().shape[0]} vs {len(initial_trg_urls)})"

        if parallel_likelihood:
            for data, initial_src_url, initial_trg_url in zip(outputs.numpy(), initial_src_urls, initial_trg_urls):
                likelihood = data[0] if regression else data[1] # parallel

                if likelihood >= threshold:
                    print(f"{likelihood:.4f}\t{initial_src_url}\t{initial_trg_url}")
        else:
            for argmax, initial_src_url, initial_trg_url in zip(outputs_argmax, initial_src_urls, initial_trg_urls):
                print(f"{'parallel' if argmax == 1 else 'non-parallel'}\t{initial_src_url}\t{initial_trg_url}")

@torch.no_grad()
def non_interactive_inference(model, tokenizer, batch_size, max_length_tokens, device, amp_context_manager,
                              src_urls, trg_urls, remove_authority=False, remove_positional_data_from_resource=False,
                              parallel_likelihood=False, threshold=-np.inf, url_separator=' ', lower=True):
    model.eval()
    results = []

    # Process URLs
    src_urls = [src_url.replace('\t', ' ') for src_url in src_urls]
    trg_urls = [trg_url.replace('\t', ' ') for trg_url in trg_urls]
    str_urls = [f"{src_url}\t{trg_url}" for src_url, trg_url in zip(src_urls, trg_urls)]
    target_urls = next(utils.tokenize_batch_from_fd(str_urls, tokenizer, batch_size,
                            f=lambda u: preprocess.preprocess_url(u, remove_protocol_and_authority=remove_authority,
                                                                  remove_positional_data=remove_positional_data_from_resource,
                                                                  separator=url_separator, lower=lower)))

    # Tokens
    tokens = utils.encode(tokenizer, target_urls, max_length_tokens)
    urls = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with amp_context_manager:
        outputs = model(urls, attention_mask).logits

    regression = outputs.cpu().detach().numpy().shape[1] == 1

    if regression:
        # Regression
        outputs = torch.sigmoid(outputs).detach()
        outputs_argmax = torch.round(outputs).squeeze(1).cpu().numpy().astype(np.int64)
    else:
        # Binary classification
        outputs = outputs.detach()
        outputs_argmax = torch.argmax(F.softmax(outputs, dim=1), dim=1).cpu().numpy()

    outputs = outputs.cpu()

    # TODO TEST_BEFORE if we use outputs_argmax.squeeze(1) instead of outputs_argmax.squeeze(), the following condition should be safe to be removed
    if len(outputs_argmax.shape) == 0:
        outputs_argmax = np.array([outputs_argmax])

    assert outputs.numpy().shape[0] == len(src_urls), "Output samples does not match with the length of src URLs " \
                                                      f"({outputs.numpy().shape[0]} vs {len(src_urls)})"
    assert outputs.numpy().shape[0] == len(trg_urls), "Output samples does not match with the length of trg URLs " \
                                                      f"({outputs.numpy().shape[0]} vs {len(trg_urls)})"

    if parallel_likelihood:
        results = [data[0] if regression else data[1] for data in outputs.numpy()]
        results = [likelihood for likelihood in results if likelihood >= threshold]
    else:
        results = ['parallel' if argmax == 1 else 'non-parallel' for argmax in outputs_argmax]

    return results
