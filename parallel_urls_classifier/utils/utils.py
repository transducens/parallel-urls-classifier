
import os
import copy
import psutil
import logging
import gzip
import lzma
from contextlib import contextmanager
import urllib.parse
import argparse

logging.getLogger("urllib3").setLevel(logging.WARNING)

def wc_l(fd, do_not_count_empty=True):
    no_lines = 0
    tell = fd.tell()

    for line in fd:
        if do_not_count_empty and line.strip() == '':
            continue

        no_lines += 1

    # Restore initial status of the fd
    fd.seek(tell)

    return no_lines

def get_layer_from_model(layer, name=None, deepcopy=True):
    could_get_layer = False

    # Get layer from model (we need to do it with a for loop since it is a generator which cannot be accessed with idx)
    for last_layer_name, last_layer_param in layer.named_parameters():
        if last_layer_name == name:
            could_get_layer = True

            break

    if name is not None:
        assert could_get_layer, f"Could not get the layer '{name}'"

    last_layer_param_data = last_layer_param.data

    if deepcopy:
        # Return a deepcopy instead of the value itself to avoid affect the model if modified

        return copy.deepcopy(last_layer_param_data)

    return last_layer_param_data

def encode(tokenizer, text, max_length=512):
    encoder = tokenizer.batch_encode_plus if isinstance(text, list) else tokenizer.encode_plus

    return encoder(text, add_special_tokens=True, truncation=True, padding="max_length",
                   return_attention_mask=True, return_tensors="pt", max_length=max_length)

def apply_model(model, tokenizer, tokens, encode=False):
    if encode:
        tokens = encode(tokenizer, tokens)

        output = model(**tokens)
    else:
        output = model(tokens)

    #input_ids = tokenized["input_ids"]
    #token_type_ids = tokenized["token_type_ids"]
    #attention_mask = tokenized["attention_mask"]

    #sentence_length = torch.count_nonzero(attention_mask).to("cpu").numpy()
    #tokens = input_ids[0,:sentence_length]

    return output

def tokenize_batch_from_fd(fd, tokenizer, batch_size, f=None, return_urls=False, add_symmetric_samples=False):
    urls = []
    initial_urls = []

    for url in fd:
        url = url.strip().split('\t')

        assert len(url) == 2, f"It was expected 2 URLs per line URLs"

        if f:
            src_url = f(url[0])
            trg_url = f(url[1])

            if isinstance(src_url, list):
                if len(src_url) != 1:
                    raise Exception(f"Unexpected size of list after applying function to URL: {len(src_url)}")

                src_url = src_url[0]
            if isinstance(trg_url, list):
                if len(trg_url) != 1:
                    raise Exception(f"Unexpected size of list after applying function to URL: {len(trg_url)}")

                trg_url = trg_url[0]
        else:
            src_url = url[0]
            trg_url = url[1]

        urls.append(f"{src_url}{tokenizer.sep_token}{trg_url}") # We don't need to add [CLS] and final [SEP]
                                                                #  (or other special tokens) since they are automatically added
        initial_urls.append((url[0], url[1]))

        if add_symmetric_samples:
            urls.append(f"{trg_url}{tokenizer.sep_token}{src_url}")
            initial_urls.append((url[1], url[0]))

        if len(urls) >= batch_size:
            if return_urls:
                yield urls, initial_urls
            else:
                yield urls

            urls = []
            initial_urls = []

    if len(urls) != 0:
        if return_urls:
            yield urls, initial_urls
        else:
            yield urls

        urls = []
        initial_urls = []

def get_current_allocated_memory_size():
    process = psutil.Process(os.getpid())
    size_in_bytes = process.memory_info().rss

    return size_in_bytes

def set_up_logging_logger(logger, filename=None, level=logging.INFO, format="[%(asctime)s] [%(name)s] [%(levelname)s] [%(module)s:%(lineno)d] %(message)s",
                          display_when_file=False):
    handlers = [
        logging.StreamHandler()
    ]

    if filename is not None:
        if display_when_file:
            # Logging messages will be stored and displayed
            handlers.append(logging.FileHandler(filename))
        else:
            # Logging messages will be stored and not displayed
            handlers[0] = logging.FileHandler(filename)

    formatter = logging.Formatter(format)

    for h in handlers:
        h.setFormatter(formatter)

        logger.addHandler(h)

    logger.setLevel(level)

    return logger

def append_from_tuple(*tuples):
    """We expect tuples where:
        - First element: list
        - Second element: value to be inserted in the list of the first component of the tuple
    """
    for l, v in tuples:
        l.append(v)

@contextmanager
def open_xz_or_gzip_or_plain(file_path, mode='rt'):
    f = None
    try:
        if file_path[-3:] == ".gz":
            f = gzip.open(file_path, mode)
        elif file_path[-3:] == ".xz":
            f = lzma.open(file_path, mode)
        else:
            f = open(file_path, mode)
        yield f

    except Exception:
        raise Exception("Error occurred while loading a file!")

    finally:
        if f:
            f.close()

def resolve_path(p):
    return os.path.realpath(os.path.expanduser(p)) if isinstance(p, str) else p

def exists(p, res_path=False, f=os.path.isfile):
    return f(resolve_path(p) if res_path else p) if isinstance(p, str) else False

def replace_multiple(s, replace_list, replace_char=' '):
    for original_char in replace_list:
        s = s.replace(original_char, replace_char)
    return s

def stringify_url(url, separator=' ', lower=False):
    if url[:8] == "https://":
        url = url[8:]
    elif url[:7] == "http://":
        url = url[7:]

    replace_chars = ['.', '-', '_', '=', '?', '\n', '\r', '\t']

    url = url.rstrip('/')

    if lower:
        url = url.lower()

    url = url.split('/')
    url = list(map(lambda u: replace_multiple(u, replace_chars).strip(), url))
    url = [' '.join([s for s in u.split(' ') if s != '']) for u in url] # Remove multiple ' '
    url = separator.join(url)

    return url

def set_up_logging(filename=None, level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] %(message)s",
                   display_when_file=False):
    handlers = [
        logging.StreamHandler()
    ]

    if filename is not None:
        if display_when_file:
            # Logging messages will be stored and displayed
            handlers.append(logging.FileHandler(filename))
        else:
            # Logging messages will be stored and not displayed
            handlers[0] = logging.FileHandler(filename)

    logging.basicConfig(handlers=handlers, level=level,
                        format=format)

def update_defined_variables_from_dict(d, provided_locals, smash=False):
    for v, _ in d.items():
        if v in provided_locals and not smash:
            raise Exception(f"Variable '{v}' is already defined and smash=False")

    provided_locals.update(d)

def preprocess_url(url, remove_protocol_and_authority=False, remove_positional_data=False, separator=' '):
    urls = []

    if isinstance(url, str):
        url = [url]

    for u in url:
        if remove_protocol_and_authority:
            # Remove protocol
            if u.startswith("https://"):
                u = u[8:]
            elif u.startswith("http://"):
                u = u[7:]
            else:
                d = u.find(':')
                s = u.find('/')

                if d != -1 and s != -1 and s - d == 1 and u[d:s + 2] == "://":
                    if len(u) > s + 2:
                        if u[s + 2] != '/':
                            u = u[s + 2:]
                    else:
                        u = u[s + 2:]

            # Remove authority
            s = u.find('/')

            if s != -1:
                u = u[s + 1:]

        if remove_positional_data:
            # e.g. https://www.example.com/resource#position -> https://www.example.com/resource

            ur = u.split('/')
            h = ur[-1].find('#')

            if h != -1:
                ur[-1] = ur[-1][:h]

            u = '/'.join(ur)

        preprocessed_url = stringify_url(urllib.parse.unquote(u), separator=separator, lower=True)

        urls.append(preprocessed_url)

    return urls

def init_weight_and_bias(model, module):
    import torch.nn as nn

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)

        if module.bias is not None:
            module.bias.data.zero_()

    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def do_reinit(model, reinit_n_layers):
    try:
        # Re-init pooler.
        model.pooler.dense.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        model.pooler.dense.bias.data.zero_()

        for param in model.pooler.parameters():
            param.requires_grad = True
    except AttributeError:
        pass

    # Re-init last n layers.
    for n in range(reinit_n_layers):
        model.encoder.layer[-(n+1)].apply(lambda module: init_weight_and_bias(model, module))

def argparse_nargs_type(*types):
    def f(arg):
        t = types[f._invoked]

        if not isinstance(arg, t):
            type_arg = type(arg)

            try:
                arg = t(arg)
            except:
                raise argparse.ArgumentTypeError(f"Arg. #{f._invoked + 1} is not instance of {str(t)}, but {str(type_arg)}")

        f._invoked += 1
        return arg

    f._invoked = 0

    return f

def get_model_parameters_applying_llrd(model, learning_rate, weight_decay=0.01):
    # This function expects the optimizer to support weight_decay

    # Based on https://gist.github.com/peggy1502/5775b0b246ef5a64a9cf7b58cd722baf#file-readability_llrd-py from https://towardsdatascience.com/advanced-techniques-for-fine-tuning-transformers-82e4e61e16e#6196
    opt_parameters = []    # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters())

    # According to AAAMLP book by A. Thakur, we generally do not use any decay
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    lr = learning_rate

    # === Pooler and regressor ======================================================

    params_0 = [p for n,p in named_parameters if ("pooler" in n or "regressor" in n or "classifier" in n)
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if ("pooler" in n or "regressor" in n or "classifier" in n)
                and not any(nd in n for nd in no_decay)]

    head_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
    opt_parameters.append(head_params)

    head_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
    opt_parameters.append(head_params)

    lr *= 0.95

    # === 12 Hidden layers ==========================================================

    for layer in range(11,-1,-1):
        params_0 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n
                    and not any(nd in n for nd in no_decay)]

        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)

        layer_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
        opt_parameters.append(layer_params)

        lr *= 0.95 # Based on https://arxiv.org/pdf/1905.05583.pdf

    # === Embeddings layer ==========================================================

    params_0 = [p for n,p in named_parameters if "embeddings" in n
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]

    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
    opt_parameters.append(embed_params)

    embed_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
    opt_parameters.append(embed_params)

    return opt_parameters

def get_tuple_if_is_not_tuple(obj):
    if not isinstance(obj, tuple):
        return (obj,)

    return obj
