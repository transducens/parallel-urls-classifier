
import re
import logging
import urllib.parse

import parallel_urls_classifier.utils.utils as utils
from parallel_urls_classifier.tokenizer import tokenize

logging.getLogger("urllib3").setLevel(logging.WARNING)

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
    url = list(map(lambda u: utils.replace_multiple(u, replace_chars).strip(), url))
    url = [' '.join([s for s in u.split(' ') if s != '']) for u in url] # Remove multiple ' '
    url = separator.join(url)

    return url

def preprocess_url(url, remove_protocol_and_authority=False, remove_positional_data=False, separator=' ',
                   stringify_instead_of_tokenization=False, remove_protocol=True):
    urls = []

    if isinstance(url, str):
        url = [url]

    if remove_protocol_and_authority:
        if not remove_protocol:
            logging.warning("'remove_protocol' is not True, but since 'remove_protocol_and_authority' is True, it will enabled")

        remove_protocol = True

    for u in url:
        u = u.rstrip('/')

        if remove_protocol:
            if u.startswith("https://"):
                u = u[8:]
            elif u.startswith("http://"):
                u = u[7:]
            else:
                # Check for other protocols different from HTTP

                d = u.find(':')
                s = u.find('/')

                if d != -1 and s != -1 and s - d == 1 and u[d:s + 2] == "://":
                    if len(u) > s + 2:
                        if u[s + 2] != '/':
                            u = u[s + 2:]
                    else:
                        u = u[s + 2:]

        if remove_protocol_and_authority:
            # The protocol should have been removed once reached this point

            s = u.find('/')

            if s == -1:
                u = "" # No resource
            else:
                u = u[s + 1:]

        if remove_positional_data:
            # e.g. https://www.example.com/resource#position -> https://www.example.com/resource

            ur = u.split('/')
            h = ur[-1].find('#')

            if h != -1:
                ur[-1] = ur[-1][:h]

            u = '/'.join(ur)

        u = urllib.parse.unquote(u)
        u = u.lower()

        # TODO TBD stringify instead of tokenize or stringify after tokenization
        if stringify_instead_of_tokenization:
            u = stringify_url(u, separator=separator)
        else:
            u = u.replace('/', separator)
            u = re.sub(r'\s+', r' ', u)
            u = ' '.join(tokenize(u))

        urls.append(u)

    return urls


