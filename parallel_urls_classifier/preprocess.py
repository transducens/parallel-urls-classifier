
import re
import logging
import urllib.parse

import parallel_urls_classifier.utils.utils as utils
from parallel_urls_classifier.tokenizer import tokenize

logging.getLogger("urllib3").setLevel(logging.WARNING)

def stringify_url(url, separator=' ', lower=False):
    url = url[utils.get_idx_after_protocol(url):]
    replace_chars = ['.', '-', '_', '=', '?', '\n', '\r', '\t']
    url = url.rstrip('/')

    if lower:
        url = url.lower()

    url = url.split('/')
    url = list(map(lambda u: utils.replace_multiple(u, replace_chars).strip(), url))
    url = [' '.join([s for s in u.split(' ') if s != '']) for u in url] # Remove multiple ' '
    url = separator.join(url)

    return url

def preprocess_url(url, remove_protocol_and_authority=False, remove_positional_data=False, separator=' ',
                   stringify_instead_of_tokenization=False, remove_protocol=True, lower=False):
    urls = []

    if isinstance(url, str):
        url = [url]

    if remove_protocol_and_authority:
        if not remove_protocol:
            logging.warning("'remove_protocol' is not True, but since 'remove_protocol_and_authority' is True, it will enabled")

        remove_protocol = True # Just for logic, but it will have no effect

    for u in url:
        u = u.rstrip('/')

        if remove_protocol_and_authority:
            u = u[get_idx_resource(u):]
        elif remove_protocol:
            u = u[utils.get_idx_after_protocol(u):]

        if remove_positional_data:
            # e.g. https://www.example.com/resource#position -> https://www.example.com/resource

            ur = u.split('/')
            h = ur[-1].find('#')

            if h != -1:
                ur[-1] = ur[-1][:h]

            u = '/'.join(ur)

        u = urllib.parse.unquote(u, errors="backslashreplace") # WARNING! It is necessary to replace, at least, \t

        if lower:
            u = u.lower()

        # TODO TBD stringify instead of tokenize or stringify after tokenization
        if stringify_instead_of_tokenization:
            u = stringify_url(u, separator=separator, lower=lower)
        else:
            u = u.replace('/', separator)
            u = re.sub(r'\s+', r' ', u)
            u = ' '.join(tokenize(u))

        urls.append(u)

    return urls


