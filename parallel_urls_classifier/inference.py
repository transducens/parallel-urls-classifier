
import sys
import random
import logging

import parallel_urls_classifier.utils.utils as utils
from parallel_urls_classifier.metrics import (
    get_metrics,
)
import parallel_urls_classifier.preprocess as preprocess

import numpy as np
import torch
import torch.nn.functional as F
import transformers

logger = logging.getLogger("parallel_urls_classifier")
logger_tokens = logging.getLogger("parallel_urls_classifier.tokens")

def inference_with_heads(model, tasks, tokenizer, inputs_and_outputs, amp_context_manager,
                         tasks_weights=None, criteria=None):
    results = {"_internal": {"total_loss": None}}

    # Inputs and outputs
    labels = {}
    urls = inputs_and_outputs["urls"]
    device = urls.device
    attention_mask = inputs_and_outputs["attention_mask"]

    if criteria:
        labels["urls_classification"] = inputs_and_outputs["labels"]

    with amp_context_manager:
        model_outputs = None

        if "mlm" in tasks:
            # "Mask" labels and mask URLs
            #  https://discuss.huggingface.co/t/bertformaskedlm-s-loss-and-scores-how-the-loss-is-computed/607/2
            _urls, _labels = transformers.DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)\
                                .torch_mask_tokens(urls.clone().cpu().detach())
            _urls = _urls.to(device)
            urls = _urls

            if criteria:
                labels["mlm"] = _labels.to(device)

        for head_task in tasks:
            model_outputs = model(head_task, urls, attention_mask) # TODO can we avoid to run the base model multiple times if we have common input?
            outputs = model_outputs.logits # Get head result
            criterion = criteria[head_task] if criteria else None
            loss_weight = tasks_weights[head_task] if tasks_weights else 1.0

            if head_task == "urls_classification":
                # Main task

                regression = outputs.cpu().detach().numpy().shape[1] == 1

                if regression:
                    # Regression
                    outputs = torch.sigmoid(outputs).squeeze(1)
                    outputs_argmax = torch.round(outputs).type(torch.int64).cpu() # Workaround for https://github.com/pytorch/pytorch/issues/54774
                else:
                    # Binary classification
                    outputs_argmax = torch.argmax(F.softmax(outputs, dim=1).cpu(), dim=1)

                if criterion:
                    loss = criterion(outputs, labels[head_task])
                    loss *= loss_weight

                results["urls_classification"] = {
                    "outputs": outputs,
                    "outputs_argmax": outputs_argmax,
                    "loss_detach": loss.cpu().detach() if criterion else None,
                    "regression": regression,
                }
            elif head_task == "mlm":
                if criterion:
                    loss = criterion(outputs.view(-1, tokenizer.vocab_size), labels[head_task].view(-1))
                    loss *= loss_weight

                results["mlm"] = {
                    "outputs": outputs,
                    "loss_detach": loss.cpu().detach() if criterion else None,
                }
            else:
                raise Exception(f"Unknown head task: {head_task}")

            # Sum loss (we don't want to define multiple losses at the same time in order to avoid high memory allocation)
            if criterion:
                if not results["_internal"]["total_loss"]:
                    results["_internal"]["total_loss"] = loss
                else:
                    results["_internal"]["total_loss"] += loss
            else:
                results["_internal"]["total_loss"] = None

    return results

@torch.no_grad()
def inference(model, block_size, batch_size, tasks, tokenizer, criteria, dataloader, device,
              amp_context_manager, classes=2):
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
    total_blocks_per_batch = max(int(np.ceil(batch_size / block_size)), 1)

    for idx, batch in enumerate(dataloader):
        for inputs_and_outputs in utils.get_data_from_batch(batch, block_size, device):
            labels = inputs_and_outputs["labels"]

            # Inference
            results = inference_with_heads(model, tasks, tokenizer, inputs_and_outputs, amp_context_manager, criteria=criteria)

            # Tasks
            loss_urls_classification = results["urls_classification"]["loss_detach"] # TODO propagate somehow? Statistics?
            outputs_argmax = results["urls_classification"]["outputs_argmax"]
            regression = results["urls_classification"]["regression"]

            if "mlm" in results:
                # TODO propagate somehow? Statistics?
                loss_mlm = results["mlm"]["loss_detach"]
                outputs_mlm = results["mlm"]["outputs"].cpu()

            loss = results["_internal"]["total_loss"].cpu()
            loss = loss.cpu() / total_blocks_per_batch
            labels = labels.cpu()

            if regression:
                labels = torch.round(labels).type(torch.long)

            total_loss += loss

            all_outputs.extend(outputs_argmax.tolist())
            all_labels.extend(labels.tolist())

    all_outputs = torch.as_tensor(all_outputs)
    all_labels = torch.as_tensor(all_labels)
    metrics = get_metrics(all_outputs, all_labels, len(all_labels), classes=classes)

    total_loss /= idx + 1
    total_acc += metrics["acc"]
    total_acc_per_class += metrics["acc_per_class"]
    total_acc_per_class_abs_precision += metrics["precision"]
    total_acc_per_class_abs_recall += metrics["recall"]
    total_acc_per_class_abs_f1 += metrics["f1"]
    total_macro_f1 += metrics["macro_f1"]

    return {
        "loss": total_loss,
        "acc": total_acc,
        "acc_per_class": total_acc_per_class,
        "precision": total_acc_per_class_abs_precision,
        "recall": total_acc_per_class_abs_recall,
        "f1": total_acc_per_class_abs_f1,
        "macro_f1": total_macro_f1,
    }

@torch.no_grad()
def interactive_inference(model, tokenizer, batch_size, max_length_tokens, device, amp_context_manager,
                          inference_from_stdin=False, remove_authority=False, remove_positional_data_from_resource=False,
                          parallel_likelihood=False, threshold=-np.inf, url_separator=' ', lower=True):
    logger.info("Inference mode enabled")

    if not inference_from_stdin:
        logger.info("Insert 2 blank lines in order to end")

    logger_tokens.debug("preprocessed_urls\tmodel_input\ttokens\ttokens2str\tunk_chars\t"
                        "initial_tokens_vs_detokenized\tinitial_tokens_vs_detokenized_len_1")

    model.eval()

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
        urls_tokens = urls.cpu()

        for idx, ut in enumerate(urls_tokens):
            url_tokens = ut[ut != tokenizer.pad_token_id]
            original_str_from_tokens = tokenizer.decode(url_tokens) # Detokenize
            str_from_tokens = '<tok_sep>'.join([tokenizer.decode(t) for t in url_tokens]) # Detokenize adding a mark between tokens
            ## Unk
            unk = torch.sum((url_tokens == tokenizer.unk_token_id).int()) # Unk tokens (this should happen just with very strange chars)
            sp_unk_vs_tokens_len = f"{len(original_str_from_tokens.split(url_separator))} vs " \
                                   f"{len(str_from_tokens.split(url_separator))}"
            sp_unk_vs_one_len_tokens = f"{sum(map(lambda u: 1 if len(u) == 1 else 0, original_str_from_tokens.split(url_separator)))} vs " \
                                       f"{sum(map(lambda u: 1 if len(u) == 1 else 0, str_from_tokens.split(url_separator)))}"

            logger_tokens.debug("%s\t%s\t%s\t%s\t%d\t%s\t%s", target_urls[idx], original_str_from_tokens,
                                                              str(url_tokens).replace('\n', ' '), str_from_tokens, unk,
                                                              sp_unk_vs_tokens_len, sp_unk_vs_one_len_tokens)

        # Inference
        results = inference_with_heads(model, ["urls_classification"], tokenizer, {"urls": urls, "attention_mask": attention_mask},
                                       amp_context_manager)

        # Get results only for main task
        outputs = results["urls_classification"]["outputs"].cpu()
        outputs_argmax = results["urls_classification"]["outputs_argmax"]
        regression = results["urls_classification"]["regression"]

        #if len(outputs_argmax.shape) == 0:
        #    outputs_argmax = np.array([outputs_argmax])

        assert outputs.numpy().shape[0] == len(initial_src_urls), "Output samples does not match with the length of src URLs " \
                                                                  f"({outputs.numpy().shape[0]} vs {len(initial_src_urls)})"
        assert outputs.numpy().shape[0] == len(initial_trg_urls), "Output samples does not match with the length of trg URLs " \
                                                                  f"({outputs.numpy().shape[0]} vs {len(initial_trg_urls)})"

        if parallel_likelihood:
            for data, initial_src_url, initial_trg_url in zip(outputs.numpy(), initial_src_urls, initial_trg_urls):
                likelihood = data if regression else data[1] # parallel

                if likelihood >= threshold:
                    print(f"{likelihood:.4f}\t{initial_src_url}\t{initial_trg_url}")
        else:
            for argmax, initial_src_url, initial_trg_url in zip(outputs_argmax, initial_src_urls, initial_trg_urls):
                print(f"{'parallel' if argmax == 1 else 'non-parallel'}\t{initial_src_url}\t{initial_trg_url}")

@torch.no_grad()
def non_interactive_inference(model, tokenizer, batch_size, max_length_tokens, device, amp_context_manager,
                              src_urls, trg_urls, remove_authority=False, remove_positional_data_from_resource=False,
                              parallel_likelihood=False, threshold=-np.inf, url_separator=' ', lower=False):
    model.eval()
    all_results = []

    # Process URLs
    src_urls = [src_url.replace('\t', ' ') for src_url in src_urls]
    trg_urls = [trg_url.replace('\t', ' ') for trg_url in trg_urls]
    str_urls = [f"{src_url}\t{trg_url}" for src_url, trg_url in zip(src_urls, trg_urls)]
    urls_generator = utils.tokenize_batch_from_fd(str_urls, tokenizer, batch_size,
                            f=lambda u: preprocess.preprocess_url(u, remove_protocol_and_authority=remove_authority,
                                                                  remove_positional_data=remove_positional_data_from_resource,
                                                                  separator=url_separator, lower=lower))

    for target_urls in urls_generator:
        # Tokens
        tokens = utils.encode(tokenizer, target_urls, max_length_tokens)
        urls = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        # Inference
        results = inference_with_heads(model, ["urls_classification"], tokenizer, {"urls": urls, "attention_mask": attention_mask},
                                    amp_context_manager)

        # Get results only for main task
        outputs = results["urls_classification"]["outputs"].cpu()
        outputs_argmax = results["urls_classification"]["outputs_argmax"]
        regression = results["urls_classification"]["regression"]

        #if len(outputs_argmax.shape) == 0:
        #    outputs_argmax = np.array([outputs_argmax])

        assert outputs.numpy().shape[0] == len(src_urls), "Output samples does not match with the length of src URLs " \
                                                        f"({outputs.numpy().shape[0]} vs {len(src_urls)})"
        assert outputs.numpy().shape[0] == len(trg_urls), "Output samples does not match with the length of trg URLs " \
                                                        f"({outputs.numpy().shape[0]} vs {len(trg_urls)})"

        if parallel_likelihood:
            _results = [data if regression else data[1] for data in outputs.numpy()]
            _results = [likelihood for likelihood in _results if likelihood >= threshold]
        else:
            _results = ['parallel' if argmax == 1 else 'non-parallel' for argmax in outputs_argmax]

        all_results.extend(_results)

    return all_results
