
import os
import sys
import time
import random
import logging
import argparse
import tempfile
import contextlib
from datetime import datetime

import parallel_urls_classifier.utils.utils as utils
from parallel_urls_classifier.inference import (
    inference,
    interactive_inference,
    inference_with_heads,
)
from parallel_urls_classifier.metrics import (
    get_metrics,
    plot_statistics,
)
import parallel_urls_classifier.preprocess as preprocess
from parallel_urls_classifier.multitask_model import MultitaskModel
import parallel_urls_classifier.dataset as dataset

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CyclicLR, LambdaLR
from torch.optim import Adam, AdamW
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    WeightedRandomSampler,
    SequentialSampler,
)
import transformers
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import sklearn

# Disable (less verbose) 3rd party logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

# Logging
logger = logging.getLogger("parallel_urls_classifier")
logger_verbose = {"tokens": logging}

def get_lr_scheduler(scheduler, optimizer, *args, **kwargs):
    scheduler_instance = None
    mandatory_args = ""

    def check_args(num_args, str_args):
        if len(args) != num_args:
            raise Exception(f"LR scheduler: '{scheduler}' mandatory args: {str_args}")

    if scheduler == "linear":
        mandatory_args = "num_warmup_steps, num_training_steps"

        check_args(2, mandatory_args)

        scheduler_instance = get_linear_schedule_with_warmup(optimizer, *args, **kwargs)
    elif scheduler == "CLR": # CyclicLR
        mandatory_args = "base_lr, max_lr"

        check_args(2, mandatory_args)

        scheduler_instance = CyclicLR(optimizer, *args, **kwargs)
    elif scheduler == "inverse_sqrt":
        mandatory_args = "num_warmup_steps"

        check_args(1, mandatory_args)

        def inverse_sqrt(current_step):
            num_warmup_steps = args[0]

            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))

            # From https://fairseq.readthedocs.io/en/latest/_modules/fairseq/optim/lr_scheduler/inverse_square_root_schedule.html
            initial_lr = optimizer.defaults["lr"]
            decay_factor = initial_lr * num_warmup_steps**0.5
            lr = decay_factor * current_step**-0.5

            return lr / initial_lr

        scheduler_instance = LambdaLR(optimizer, inverse_sqrt, **kwargs)
    else:
        raise Exception(f"Unknown LR scheduler: {scheduler}")

    logger.debug("LR scheduler: '%s' mandatory args: %s: %s", scheduler, mandatory_args, str(args))
    logger.debug("LR scheduler: '%s' optional args: %s", scheduler, str(kwargs))

    return scheduler_instance

def load_model(tasks, tasks_kwargs, model_input="", pretrained_model="", device=""):
    if len(tasks) == 0:
        raise Exception("At least 1 head is mandatory")
    if set(tasks) != set(tasks_kwargs):
        raise Exception("Different tasks provided to 'tasks' and 'tasks_kwargs': "
                        f"{set(tasks)} vs {set(tasks_kwargs)}")

    multitask_model = MultitaskModel.create(pretrained_model, tasks, tasks_kwargs)

    if model_input:
        logger.info("Loading model: '%s'", model_input)

        multitask_model.from_pretrained_wrapper(model_input, device=device)
    else:
        # Move model to device
        if device:
            multitask_model = multitask_model.to(device)

    return multitask_model

def load_tokenizer(pretrained_model):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    return tokenizer

def get_amp_context_manager(cuda_amp, use_cuda):
    use_cuda = torch.cuda.is_available()
    amp_context_manager = contextlib.nullcontext()
    amp_grad_scaler = None
    _cuda_amp = cuda_amp

    # Configure AMP context manager
    if cuda_amp and use_cuda:
        amp_context_manager = torch.cuda.amp.autocast()
        amp_grad_scaler = torch.cuda.amp.GradScaler()
        _cuda_amp = True

        logger.debug("AMP enabled for CUDA")
    elif cuda_amp:
        _cuda_amp = False
        logger.warning("AMP could not be enabled")

    return amp_context_manager, amp_grad_scaler, _cuda_amp

# TODO TBD use https://pypi.org/project/imbalanced-learn/ for unbalanced data instead of custom implementation

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
    block_size = args.block_size
    max_tokens = args.max_tokens if args.max_tokens > 0 else None
    epochs = args.epochs # BE AWARE! "epochs" might be fake due to --train-until-patience
    force_cpu = args.force_cpu
    use_cuda = utils.use_cuda(force_cpu=force_cpu) # Will be True if possible and False otherwise
    device = torch.device("cuda:0" if use_cuda else "cpu")
    is_device_gpu = device.type.startswith("cuda")
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
    remove_positional_data_from_resource = args.remove_positional_data_from_resource
    add_symmetric_samples = args.add_symmetric_samples
    log_directory = args.log_directory
    regression = args.regression
    train_until_patience = args.train_until_patience
    url_separator = args.url_separator
    url_separator_new_token = args.url_separator_new_token
    learning_rate = args.learning_rate
    re_initialize_last_n_layers = max(0, args.re_initialize_last_n_layers)
    scheduler_str = args.lr_scheduler
    lr_scheduler_args_linear = utils.get_tuple_if_is_not_tuple(args.lr_scheduler_args_linear)
    lr_scheduler_args_clr = args.lr_scheduler_args_clr
    lr_scheduler_args_inverse_sqrt = utils.get_tuple_if_is_not_tuple(args.lr_scheduler_args_inverse_sqrt)
    cuda_amp = args.cuda_amp
    llrd = args.llrd
    lock_file = args.lock_file
    stringify_instead_of_tokenization = args.stringify_instead_of_tokenization
    lower = args.lowercase
    auxiliary_tasks = args.auxiliary_tasks if args.auxiliary_tasks else []
    auxiliary_tasks_weights = args.auxiliary_tasks_weights
    freeze_embeddings_layer = args.freeze_embeddings_layer
    waiting_time = args.waiting_time
    remove_instead_of_truncate = args.remove_instead_of_truncate

    if auxiliary_tasks:
        _auxiliary_tasks_weights = {}

        if not auxiliary_tasks_weights:
            for task in auxiliary_tasks:
                _auxiliary_tasks_weights[task] = 1.0
        elif len(auxiliary_tasks) != len(auxiliary_tasks_weights):
            raise Exception("You need to provide weights either for all the auxiliary tasks or for none of them")
        else:
            for task, weight in zip(auxiliary_tasks, auxiliary_tasks_weights):
                _auxiliary_tasks_weights[task] = weight

        auxiliary_tasks_weights = _auxiliary_tasks_weights
        auxiliary_tasks_weights["urls_classification"] = 1.0

        logger.debug("Auxiliary tasks weights: %s", str(auxiliary_tasks_weights))

    auxiliary_tasks = sorted(list(set(utils.get_tuple_if_is_not_tuple(auxiliary_tasks))))
    all_tasks = ["urls_classification"] + auxiliary_tasks

    if not block_size:
        block_size = batch_size
    if batch_size < block_size:
        logger.warning("Block size has to be less than or equal to batch size: updating block size to batch size: %d", batch_size)

        block_size = batch_size

    if lock_file and utils.exists(lock_file):
        logger.warning("Lock file ('%s') exists: finishing training", lock_file)

        sys.exit(0)
    if lock_file:
        logger.debug("Lock file will be created if the training finishes: %s", lock_file)

    num_labels = 1 if regression else 2
    classes = 2
    amp_context_manager, amp_grad_scaler, cuda_amp = get_amp_context_manager(cuda_amp, use_cuda)

    if scheduler_str in ("linear",) and train_until_patience:
        # Depending on the LR scheduler, the training might even stop at some point (e.g. linear LR scheduler will set the LR=0 if the run epochs is greater than the provided epochs)
        logger.warning("You set a LR scheduler ('%s' scheduler) which conflicts with --train-until-patince: you might want to check this out and change the configuration", scheduler_str)

    # Enable cuDNN benchmark
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    # Disable parallelism since throws warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    if apply_inference and not model_input:
        logger.warning("Flag --model-input is recommended when --inference is provided: waiting %d seconds before proceed", waiting_time)

        time.sleep(waiting_time)

    logger.debug("Pretrained model architecture: %s", pretrained_model)

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
    if max_tokens and max_tokens < max_length_tokens:
        logger.warning("The specified max_tokens has to be greater or equal that the max length tokens of the model:"
                       "changing value from %d to %d", max_tokens, max_length_tokens)

        max_tokens = max_length_tokens

    logger.info("Device: %s", device)

    if not apply_inference:
        logger.debug("Train URLs file (parallel, non-parallel): (%s, %s)", file_parallel_urls_train, file_non_parallel_urls_train)
        logger.debug("Dev URLs file (parallel, non-parallel): (%s, %s)", file_parallel_urls_dev, file_non_parallel_urls_dev)
        logger.debug("Test URLs file (parallel, non-parallel): (%s, %s)", file_parallel_urls_test, file_non_parallel_urls_test)

    all_tasks_kwargs = {}
    all_tasks_kwargs["urls_classification"] = {"num_labels": num_labels}
    total_auxiliary_tasks = 0

    if "mlm" in auxiliary_tasks:
        all_tasks_kwargs["mlm"] = {}

        logger.info("Using auxiliary task: mlm")

        total_auxiliary_tasks += 1

    if total_auxiliary_tasks == 0:
        logger.info("Not using any auxiliary task")

    if total_auxiliary_tasks != len(auxiliary_tasks):
        # We forgot something (e.g. update the code according a new auxiliary tasks)
        raise Exception("The specified auxiliary tasks could not be loaded (bug): "
                        f"{' '.join(auxiliary_tasks)} ({len(auxiliary_tasks)})")

    model = load_model(all_tasks, all_tasks_kwargs, model_input=model_input, pretrained_model=pretrained_model, device=device)

    if model_output:
        model_tasks = model.get_tasks_names()
        two_or_more_tasks = len(model_tasks) > 1
        logger.info("Model will be stored: '%s.heads.%s%s%s'", model_output, '{' if two_or_more_tasks else '',
                                                               ','.join(model_tasks), '}' if two_or_more_tasks else '')

        all_output_paths = [model.__class__.get_task_model_path(model_output, t) for t in model_tasks]
        _wait = False

        for output_path in all_output_paths:
            if utils.exists(output_path, f=os.path.isdir):
                if args.overwrite_output_model:
                    logger.warning("Provided output model does exist (file: '%s'): it will be updated: waiting %d seconds before proceed",
                                    output_path, waiting_time)

                    _wait = True
                else:
                    raise Exception(f"Provided output model does exist: '{output_path}'")

        if _wait:
            time.sleep(waiting_time)

    tokenizer = load_tokenizer(pretrained_model)
    fine_tuning = not args.do_not_fine_tune
    model_embeddings_size = model.get_base_model().base_model.embeddings.word_embeddings.weight.shape[0]

    if url_separator_new_token:
        # Add new special token (URL separator)
        num_added_toks = tokenizer.add_tokens([url_separator], special_tokens=True)

        logger.debug("New tokens added to tokenizer: %d", num_added_toks)

        if not model_input:
            model.get_base_model().resize_token_embeddings(len(tokenizer))

            if freeze_embeddings_layer:
                logger.warning("Embeddings layer is frozen, and new tokens will not be trained")
        elif model_embeddings_size + 1 == len(tokenizer):
            logger.warning("You've loaded a model which does not have the new token, so the results might be unexpected")

            model.get_base_model().resize_token_embeddings(len(tokenizer))

    model_embeddings_size = model.get_base_model().base_model.embeddings.word_embeddings.weight.shape[0]

    if model_embeddings_size != len(tokenizer):
        logger.error("Embedding layer size does not match with the tokenizer size: %d vs %d", model_embeddings_size, len(tokenizer))

    if apply_inference:
        interactive_inference(model, tokenizer, batch_size, max_length_tokens, device, amp_context_manager,
                              inference_from_stdin=inference_from_stdin, remove_authority=remove_authority,
                              parallel_likelihood=parallel_likelihood, threshold=threshold, url_separator=url_separator,
                              remove_positional_data_from_resource=remove_positional_data_from_resource, lower=lower)

        # Stop execution
        return

    if regression:
        if imbalanced_strategy == "weighted-loss":
            logger.warning("Incompatible weight strategy ('%s'): regression can't be applied with the selected strategy: "
                           "it will not be applied", imbalanced_strategy)

            imbalanced_strategy = "none"

    # Unfreeze heads layers
    for task in all_tasks:
        head = model.get_head(task)

        for param in head.parameters():
            param.requires_grad = True

    # Freeze layers of the model, if needed
    for param in model.get_base_model().parameters():
        param.requires_grad = fine_tuning

    # Freeze embeddings layer, if needed
    for param in model.get_base_model().base_model.embeddings.parameters():
        param.requires_grad = not freeze_embeddings_layer

    last_layer_output = utils.get_layer_from_model(model.get_base_model().base_model.encoder.layer[-1], name="output.dense.weight")

    # Re-initilize last N layers from the pre-trained model
    if fine_tuning and re_initialize_last_n_layers > 0:
        utils.do_reinit(model.get_base_model().base_model, re_initialize_last_n_layers)

    logger.debug("Allocated memory before starting tokenization: %d", utils.get_current_allocated_memory_size())

    for fd, l in ((file_parallel_urls_train, parallel_urls_train), (file_non_parallel_urls_train, non_parallel_urls_train),
                  (file_parallel_urls_dev, parallel_urls_dev), (file_non_parallel_urls_dev, non_parallel_urls_dev)):
                  # We add symmetric examples in dev as well in order to be sure we get a robust model
        batch = utils.tokenize_batch_from_fd(
                    fd, tokenizer, batch_size,
                    f=lambda u: preprocess.preprocess_url(u, remove_protocol_and_authority=remove_authority,
                                                          remove_positional_data=remove_positional_data_from_resource,
                                                          separator=url_separator, lower=lower,
                                                          stringify_instead_of_tokenization=stringify_instead_of_tokenization),
                    add_symmetric_samples=add_symmetric_samples)

        for batch_urls in batch:
            l.extend(batch_urls)

    for fd, l in ((file_parallel_urls_test, parallel_urls_test), (file_non_parallel_urls_test, non_parallel_urls_test)):
        batch = utils.tokenize_batch_from_fd(
                    fd, tokenizer, batch_size,
                    f=lambda u: preprocess.preprocess_url(u, remove_protocol_and_authority=remove_authority,
                                                          remove_positional_data=remove_positional_data_from_resource,
                                                          separator=url_separator, lower=lower,
                                                          stringify_instead_of_tokenization=stringify_instead_of_tokenization))

        for batch_urls in batch:
            l.extend(batch_urls)

    logger.info("%d pairs of parallel URLs loaded (train)", len(parallel_urls_train))
    logger.info("%d pairs of non-parallel URLs loaded (train)", len(non_parallel_urls_train))
    logger.info("%d pairs of parallel URLs loaded (dev)", len(parallel_urls_dev))
    logger.info("%d pairs of non-parallel URLs loaded (dev)", len(non_parallel_urls_dev))
    logger.info("%d pairs of parallel URLs loaded (test)", len(parallel_urls_test))
    logger.info("%d pairs of non-parallel URLs loaded (test)", len(non_parallel_urls_test))

    min_train_samples = min(len(non_parallel_urls_train), len(parallel_urls_train))
    classes_count = np.array([len(non_parallel_urls_train), len(parallel_urls_train)]) # non-parallel URLs label is 0, and
                                                                                       #  parallel URLs label is 1
    min_classes_weights = min_train_samples / classes_count

    if imbalanced_strategy == "none":
        # Is the data imbalanced? If so, warn about it

        for cw in min_classes_weights:
            if cw < 0.9:
                logger.warning("Your data seems to be imbalanced and you did not selected any imbalanced data strategy")
                break

    # Datasets
    dataset_train = dataset.SmartBatchingURLsDataset(parallel_urls_train, non_parallel_urls_train, tokenizer,
                                                     max_length_tokens, regression=regression, set_desc="train",
                                                     remove_instead_of_truncate=remove_instead_of_truncate,
                                                     imbalanced_strategy=imbalanced_strategy)
    dataset_dev = dataset.SmartBatchingURLsDataset(parallel_urls_dev, non_parallel_urls_dev, tokenizer,
                                                   max_length_tokens, regression=regression, set_desc="dev")
    dataset_test = dataset.SmartBatchingURLsDataset(parallel_urls_test, non_parallel_urls_test, tokenizer,
                                                    max_length_tokens, regression=regression, set_desc="test")

    logger.debug("Total tokens (train): %d", dataset_train.total_tokens)
    logger.debug("Total tokens (dev): %d", dataset_dev.total_tokens)
    logger.debug("Total tokens (test): %d", dataset_test.total_tokens)

    dataloader_train = dataset_train.get_dataloader(batch_size, device, force_cpu, args.dataset_workers, max_tokens=max_tokens)
    dataloader_dev = dataset_dev.get_dataloader(batch_size, device, force_cpu, args.dataset_workers, max_tokens=max_tokens)
    dataloader_test = dataset_test.get_dataloader(batch_size, device, force_cpu, args.dataset_workers, max_tokens=max_tokens)

    #logger.info("Train URLs: %.2f GB", dataset_train.size_gb)
    #logger.info("Dev URLs: %.2f GB", dataset_dev.size_gb)
    #logger.info("Test URLs: %.2f GB", dataset_test.size_gb)

    classes_weights = torch.as_tensor(sklearn.utils.class_weight.compute_class_weight("balanced",
                                                                                      classes=np.unique(dataset_train.labels),
                                                                                      y=dataset_train.labels.numpy()),
                                      dtype=torch.float)
    loss_weight = classes_weights if imbalanced_strategy == "weighted-loss" else None
    training_steps_per_epoch = len(dataloader_train)
    training_steps = training_steps_per_epoch * epochs # BE AWARE! "epochs" might be fake due to --train-until-patience
    criteria = {}

    logger.debug("Classes weights: %s", str(classes_weights))

    # Get criterion for each head task
    for head_task in all_tasks:
        if head_task == "urls_classification":
            if regression:
                # Regression
                criterion = nn.MSELoss()
            else:
                # Binary classification
                criterion = nn.CrossEntropyLoss(weight=loss_weight, reduction="mean") # Raw input, not normalized (i.e. don't apply softmax)
                # TODO change to BCELoss? bceloss vs crossentropyloss -> BCELoss seems to fit here
        elif head_task == "mlm":
            criterion = nn.CrossEntropyLoss()
        else:
            raise Exception(f"Unknown head task: {head_task}")

        criterion = criterion.to(device)

        criteria[head_task] = criterion

    if llrd:
        #model_parameters = utils.get_model_parameters_applying_llrd(model, learning_rate, weight_decay=0.0) # Adam
        model_parameters = utils.get_model_parameters_applying_llrd(model, learning_rate, weight_decay=0.01) # AdamW
    else:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())

    #optimizer = Adam(model_parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    optimizer = AdamW(model_parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)

    # Get LR scheduler args
    if scheduler_str == "linear":
        scheduler_args = [int(lr_scheduler_args_linear[0] * training_steps), training_steps]
        scheduler_kwargs = {}
    elif scheduler_str == "CLR":
        scheduler_max_lr, scheduler_step_size, scheduler_mode, scheduler_gamma, scheduler_max_lr_factor, scheduler_step_size_factor \
            = lr_scheduler_args_clr

        if learning_rate > scheduler_max_lr:
            new_scheduler_max_lr = learning_rate * scheduler_max_lr_factor # Based on the CLR paper (possible values are [3.0, 4.0])

            logger.warning("LR scheduler: '%s': provided LR (%f) is greater than provided max. LR (%f): setting max. LR to %f",
                           scheduler_str, learning_rate, scheduler_max_lr, new_scheduler_max_lr)

            scheduler_max_lr = new_scheduler_max_lr
        if scheduler_step_size <= 0:
            scheduler_step_size = scheduler_step_size_factor * training_steps_per_epoch # Based on the CLR paper (possible values are [2, ..., 8])

            logger.warning("LR scheduler: '%s': provided step size is 0 or negative: setting value to %d", scheduler_str, scheduler_step_size)

        scheduler_args = [learning_rate, scheduler_max_lr]
        scheduler_kwargs = {"step_size_up": scheduler_step_size, "step_size_down": scheduler_step_size,
                            "mode": scheduler_mode, "gamma": scheduler_gamma,
                            "cycle_momentum": False, # https://github.com/pytorch/pytorch/issues/73910
                            }
    elif scheduler_str == "inverse_sqrt":
        scheduler_args = [int(lr_scheduler_args_inverse_sqrt[0] * training_steps)]
        scheduler_kwargs = {}
    else:
        raise Exception(f"Unknown LR scheduler: {scheduler}")

    scheduler = get_lr_scheduler(scheduler_str, optimizer, *scheduler_args, **scheduler_kwargs)

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
    total_blocks_per_batch = 1 if max_tokens else max(int(np.ceil(batch_size / block_size)), 1)

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
        total_train_tokens = 0
        total_train_tokens_with_padding = 0
        idx = -1

        model.train()

        for batch in dataloader_train:
            if max_tokens and batch is None:
                # Batch is under construction using max_tokens...
                continue

            idx += 1
            batch_outputs = []
            batch_labels = []
            loss_value = None
            tasks_loss_value = {t: 0.0 for t in all_tasks}

            #optimizer.zero_grad() # https://discuss.pytorch.org/t/model-zero-grad-or-optimizer-zero-grad/28426/6
            model.zero_grad()

            # Process in block_size blocks in order to avoid OOM errors, but use batch_size for update the model
            for inputs_and_outputs in utils.get_data_from_batch(batch, None if max_tokens else block_size, device):
                labels = inputs_and_outputs["labels"]
                total_train_tokens += sum([len(urls[urls != tokenizer.pad_token_id]) for urls in inputs_and_outputs["urls"]])
                total_train_tokens_with_padding += sum([len(urls) for urls in inputs_and_outputs["urls"]])

                # Inference
                results = inference_with_heads(model, all_tasks, tokenizer, inputs_and_outputs, amp_context_manager,
                                               criteria=criteria, tasks_weights=auxiliary_tasks_weights)

                # Main task
                outputs_argmax = results["urls_classification"]["outputs_argmax"]
                loss = results["_internal"]["total_loss"] # Multiple losses if auxiliary tasks were used
                loss /= total_blocks_per_batch # Gradient accumulation

                # Results
                if loss_value is None:
                    loss_value = loss.cpu().detach().numpy() # Accumulated loss of all tasks
                else:
                    loss_value += loss.cpu().detach().numpy()

                labels = labels.cpu()

                if regression:
                    labels = torch.round(labels).type(torch.long)

                batch_outputs.extend(outputs_argmax.tolist())
                batch_labels.extend(labels.tolist())

                if cuda_amp:
                    amp_grad_scaler.scale(loss).backward()
                else:
                    loss.backward()

                for t in results.keys():
                    if t.startswith("_"):
                        # It is not a task, but some internal value
                        continue

                    # Losses per task of a whole batch (cumulate loss of blocks per task)
                    tasks_loss_value[t] += results[t]["loss_detach"].numpy()

                    # Drop immediate buffers (https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3)
                    loss.detach_()

            if cuda_amp:
                # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping

                amp_grad_scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            batch_labels_tensor = torch.as_tensor(batch_labels)
            current_batch_size = batch_labels_tensor.reshape(-1).shape[0]

            all_outputs.extend(batch_outputs)
            all_labels.extend(batch_labels)

            # Get metrics
            log = (idx + 1) % show_statistics_every_batches == 0
            metrics = get_metrics(torch.as_tensor(batch_outputs), batch_labels_tensor, current_batch_size,
                                  classes=classes, idx=idx, log=log)

            if log:
                logger.debug("[train:batch#%d] Loss: %f", idx + 1, loss_value)
                logger.debug("[train:batch#%d] Processed tokens (without padding): %d (%d)", idx + 1, total_train_tokens_with_padding,
                             total_train_tokens)

                if len(all_tasks) > 1:
                    for t, v in tasks_loss_value.items():
                        # Log loss of all tasks
                        logger.debug("[train:batch#%d] Loss task '%s': %f", idx + 1, t, v)

            show_statistics = (epoch == 0 and idx == 0) or (idx + 1) % show_statistics_every_batches == 0
            epoch_loss += loss_value
            epoch_acc += metrics["acc"]
            epoch_acc_per_class += metrics["acc_per_class"]
            epoch_acc_per_class_abs += metrics["f1"]
            epoch_macro_f1 += metrics["macro_f1"]

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

            if show_statistics:
                # LRs statistics
                all_lrs = scheduler.get_last_lr()
                current_lr = all_lrs[0]
                len_lrs = len(all_lrs)

                if len_lrs != 1:
                    logger.debug("[batch#%d] LR scheduler: First and last LRs: %s", idx + 1, f"{str(all_lrs[0:10])[:-1]} ... {str(all_lrs[10:])[1:]}")
                else:
                    logger.debug("[batch#%d] LR scheduler: Current LR: %.8f", idx + 1, current_lr)

            if cuda_amp:
                amp_grad_scaler.step(optimizer)
                amp_grad_scaler.update()
            else:
                optimizer.step()

            scheduler.step()

        if total_train_tokens != dataset_train.total_tokens:
            if imbalanced_strategy in ("none", "weighted-loss"):
                logger.error("Total processed tokens are different from the initial total tokens: %d vs %d",
                             total_train_tokens, dataset_train.total_tokens)
            else:
                # The selected imbalanced_strategy modifies the number of samples, so we can't compare if it's what we expect
                pass

        current_last_layer_output = utils.get_layer_from_model(model.get_base_model().base_model.encoder.layer[-1], name="output.dense.weight")
        layer_updated = (current_last_layer_output != last_layer_output).any().cpu().detach().numpy()

        logger.debug("Has the model layer been updated? %s", 'yes' if layer_updated else 'no')

        all_outputs = torch.as_tensor(all_outputs)
        all_labels = torch.as_tensor(all_labels)
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

        dev_inference_metrics = inference(model, block_size, batch_size, all_tasks, tokenizer, criteria, dataset_dev,
                                          device, amp_context_manager, classes=classes, max_tokens=max_tokens)

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
                model.save_pretrained_wrapper(model_output)

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

    final_loss /= epoch
    final_acc /= epoch
    final_acc_per_class /= epoch
    final_acc_per_class_abs /= epoch
    final_macro_f1 /= epoch

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

        logger.info("Loading best model (dev score): %s", str(best_dev))

        model.from_pretrained_wrapper(model_output, device=device)

    dev_inference_metrics = inference(model, block_size, batch_size, all_tasks, tokenizer, criteria, dataset_dev,
                                      device, amp_context_manager, classes=classes, max_tokens=max_tokens)

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

    test_inference_metrics = inference(model, block_size, batch_size, all_tasks, tokenizer, criteria, dataset_test,
                                       device, amp_context_manager, classes=classes, max_tokens=max_tokens)

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

    if lock_file:
        # Create lock file since the training finished
        from pathlib import Path

        Path(lock_file).touch()

        logger.debug("Lock file created: %s", lock_file)

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Parallel URLs classifier")
    inference = "--inference" in sys.argv

    if not inference:
        parser.add_argument('parallel_urls_train_filename', type=argparse.FileType('rt', errors="backslashreplace"), help="Filename with parallel URLs (TSV format)")
        parser.add_argument('parallel_urls_dev_filename', type=argparse.FileType('rt', errors="backslashreplace"), help="Filename with parallel URLs (TSV format)")
        parser.add_argument('parallel_urls_test_filename', type=argparse.FileType('rt', errors="backslashreplace"), help="Filename with parallel URLs (TSV format)")
        parser.add_argument('non_parallel_urls_train_filename', type=argparse.FileType('rt', errors="backslashreplace"), help="Filename with non-parallel URLs (TSV format)")
        parser.add_argument('non_parallel_urls_dev_filename', type=argparse.FileType('rt', errors="backslashreplace"), help="Filename with non-parallel URLs (TSV format)")
        parser.add_argument('non_parallel_urls_test_filename', type=argparse.FileType('rt', errors="backslashreplace"), help="Filename with non-parallel URLs (TSV format)")

    parser.add_argument('--batch-size', type=int, default=16, help="Batch size. Elements which will be processed before proceed to train, but the whole batch will be processed in blocks in order to avoid OOM errors")
    parser.add_argument('--block-size', type=int, help="Block size. Elements which will be provided to the model at once")
    parser.add_argument('--max-tokens', type=int, default=-1, help="Process batches in groups tokens size (fairseq style). Batch size is still relevant since the value is used when batches are needed (e.g. sampler from dataset)")
    parser.add_argument('--epochs', type=int, default=3, help="Epochs")
    parser.add_argument('--do-not-fine-tune', action="store_true", help="Do not apply fine-tuning (default weights)")
    parser.add_argument('--dataset-workers', type=int, default=-1, help="No. workers when loading the data in the dataset. When negative, all available CPUs will be used")
    parser.add_argument('--pretrained-model', default="xlm-roberta-base", help="Pretrained model")
    parser.add_argument('--max-length-tokens', type=int, default=256, help="Max. length for the generated tokens")
    parser.add_argument('--model-input', help="Model input path which will be loaded")
    parser.add_argument('--model-output', help="Model output path where the model will be stored")
    parser.add_argument('--inference', action="store_true", help="Do not train, just apply inference (flag --model-input is recommended)")
    parser.add_argument('--inference-from-stdin', action="store_true", help="Read inference from stdin")
    parser.add_argument('--parallel-likelihood', action="store_true", help="Print parallel likelihood instead of classification string (inference)")
    parser.add_argument('--threshold', type=float, default=-np.inf, help="Only print URLs which have a parallel likelihood greater than the provided threshold (inference)")
    parser.add_argument('--imbalanced-strategy', type=str, choices=["none", "over-sampling", "weighted-loss"], default="none", help="Strategy for dealing with imbalanced data")
    parser.add_argument('--patience', type=int, default=0, help="Patience before stopping the training")
    parser.add_argument('--train-until-patience', action="store_true", help="Train until patience value is reached (--epochs will be ignored in order to stop, but will still be used for other actions like LR scheduler)")
    parser.add_argument('--do-not-load-best-model', action="store_true", help="Do not load best model for final dev and test evaluation (--model-output is necessary)")
    parser.add_argument('--overwrite-output-model', action="store_true", help="Overwrite output model if it exists (initial loading)")
    parser.add_argument('--remove-authority', action="store_true", help="Remove protocol and authority from provided URLs")
    parser.add_argument('--remove-positional-data-from-resource', action="store_true", help="Remove content after '#' in the resorce (e.g. https://www.example.com/resource#position -> https://www.example.com/resource)")
    parser.add_argument('--add-symmetric-samples', action="store_true", help="Add symmetric samples for training (if (src, trg) URL pair is provided, (trg, src) URL pair will be provided as well)")
    parser.add_argument('--force-cpu', action="store_true", help="Run on CPU (i.e. do not check if GPU is possible)")
    parser.add_argument('--log-directory', help="Directory where different log files will be stored")
    parser.add_argument('--regression', action="store_true", help="Apply regression instead of binary classification")
    parser.add_argument('--url-separator', default='/', help="Separator to use when URLs are stringified")
    parser.add_argument('--url-separator-new-token', action="store_true", help="Add special token for URL separator")
    parser.add_argument('--learning-rate', type=float, default=2e-5, help="Learning rate")
    parser.add_argument('--lr-scheduler', choices=["linear", "CLR", "inverse_sqrt"], default="CLR", help="LR scheduler")
    parser.add_argument('--lr-scheduler-args-linear', nargs=1, metavar=("warmup_steps_percentage",), default=(0.1), \
                        type=utils.argparse_nargs_type(float), help="Args. for linear scheduler")
    parser.add_argument('--lr-scheduler-args-clr', nargs=6, metavar=("max_lr", "step_size", "mode", "gamma", "max_lr_factor", "step_size_factor"), \
                        default=(8e-5, 2000, "triangular2", 1.0, 4, 2),
                        type=utils.argparse_nargs_type(float, int, str, float, {"type": int, "choices": (3, 4)}, {"type": int, "choices": tuple(range(2,8+1))}), \
                        help="Args. for CLR scheduler")
    parser.add_argument('--lr-scheduler-args-inverse-sqrt', nargs=1, metavar=("warmup_steps_percentage",), default=(0.1), \
                        type=utils.argparse_nargs_type(float), help="Args. for inverse sqrt")
    parser.add_argument('--re-initialize-last-n-layers', type=int, default=3, help="Re-initialize last N layers from pretained model (will be applied only when fine-tuning the model)")
    parser.add_argument('--cuda-amp', action="store_true", help="Use CUDA AMP (Automatic Mixed Precision)")
    parser.add_argument('--llrd', action="store_true", help="Apply LLRD (Layer-wise Learning Rate Decay)")
    parser.add_argument('--stringify-instead-of-tokenization', action="store_true", help="Preprocess URLs applying custom stringify instead of tokenization")
    parser.add_argument('--lowercase', action="store_true", help="Lowercase URLs while preprocessing")
    parser.add_argument('--auxiliary-tasks', type=str, nargs='*', choices=["mlm"], help="Tasks which will try to help to the main task (multitasking)")
    parser.add_argument('--auxiliary-tasks-weights', type=float, nargs='*', help="Weights for the loss of the auxiliary tasks. If none is provided, the weights will be 1, but if any is provided, as many weights as auxiliary tasks will have to be provided")
    parser.add_argument('--freeze-embeddings-layer', action="store_true", help="Freeze embeddings layer")
    parser.add_argument('--remove-instead-of-truncate', action="store_true", help="Remove pairs of URLs which would need to be truncated (if not enabled, truncation will be applied). This option will be only applied to the training set")

    parser.add_argument('--seed', type=int, default=71213, help="Seed in order to have deterministic results (not fully guaranteed). Set a negative number in order to disable this feature")
    parser.add_argument('--plot', action="store_true", help="Plot statistics (matplotlib pyplot) in real time")
    parser.add_argument('--plot-path', help="If set, the plot will be stored instead of displayed")
    parser.add_argument('--lock-file', help="If set, and the file does not exist, it will be created once the training finishes. If does exist, the training will not be executed")
    parser.add_argument('--waiting-time', type=int, default=20, help="Waiting time, if needed for letting the user react")


    parser.add_argument('-v', '--verbose', action="store_true", help="Verbose logging mode")

    args = parser.parse_args()

    return args

def fix_log_directory(args):
    log_directory = args.log_directory
    waiting_time = args.waiting_time

    if not log_directory:
        log_directory = tempfile.mkdtemp(prefix=f"puc_{datetime.now().strftime('%Y%m%d%H%M%S')}_")

    if not utils.exists(log_directory, f=os.path.isdir):
        raise Exception(f"Provided log directory does not exist: '{log_directory}'")
    else:
        logger.info("Log directory: %s", log_directory)

        log_directory_files = os.listdir(utils.resolve_path(log_directory))

        if len(log_directory_files) != 0:
            logger.warning("Log directory contain %d files: waiting %d seconds before proceed", len(log_directory_files), waiting_time)

            time.sleep(waiting_time)

    args.log_directory = log_directory

def cli():
    global logger
    global logger_verbose

    args = initialization()

    # Logging
    logger = utils.set_up_logging_logger(logger, level=logging.DEBUG if args.verbose else logging.INFO)

    logger.debug("Arguments processed: {}".format(str(args))) # First logging message should be the processed arguments

    # Verbose loggers
    logger_verbose["tokens"] = logging.getLogger("parallel_urls_classifier.tokens")
    logger_verbose["tokens"].propagate = False

    fix_log_directory(args) # We are going to use args.log_directory, so fix it if needed

    logger_verbose["tokens"] = utils.set_up_logging_logger(logger_verbose["tokens"], level=logging.DEBUG if args.verbose else logging.INFO,
                                                           filename=f"{args.log_directory}/tokens", format="%(asctime)s\t%(levelname)s\t%(message)s")

    main(args)

if __name__ == "__main__":
    # https://stackoverflow.com/questions/16549332/python-3-how-to-specify-stdin-encoding
    sys.stdin.reconfigure(encoding='utf-8', errors="backslashreplace")

    cli()
