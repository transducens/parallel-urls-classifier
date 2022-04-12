
import os
import copy
import psutil
import logging
import gzip
import lzma
from contextlib import contextmanager
import urllib.parse

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

def get_layer_from_model(layer, name=None):
    could_get_layer = False

    # Get layer from model (we need to do it with a for loop since it is a generator which cannot be accessed with idx)
    for last_layer_name, last_layer_param in layer.named_parameters():
        if last_layer_name == name:
            could_get_layer = True

            break

    if name is not None:
        assert could_get_layer, f"Could not get the layer '{name}'"

    # Return a deepcopy instead of the value itself to avoid affect the model if modified
    last_layer_param_data = copy.deepcopy(last_layer_param.data)

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

def stringify_url(url):
    if url[:8] == "https://":
        url = url[8:]
    elif url[:7] == "http://":
        url = url[7:]

    replace_chars = ['.', '-', '_', '=', '?', '\n', '\r', '\t']

    url = url.rstrip('/')
    url = url.split('/')
    url = list(map(lambda u: replace_multiple(u, replace_chars).strip(), url))
    url = [' '.join([s for s in u.split(' ') if s != '']) for u in url] # Remove multiple ' '
    url = ' '.join(url)

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

def preprocess_url(url, remove_protocol_and_authority=False):
    urls = []

    if isinstance(url, str):
        url = [url]

    for u in url:
        if remove_protocol_and_authority:
            ur = u.split('/')

            if ur[0] in ("http:", "https:") and ur[1] == '':
                u = '/'.join(ur[3:])

        preprocessed_url = stringify_url(urllib.parse.unquote(u)).lower()

        urls.append(preprocessed_url)

    return urls
