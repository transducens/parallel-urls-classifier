
import os
import gzip
import logging
import argparse
import itertools
from contextlib import nullcontext

import utils

def _and(a, b):
    return a and b

def _or(a, b):
    return a or b

def main(args):
    data = args.data
    prefix = args.prefix
    lang = args.lang
    mime = args.mime
    gold_standard = args.gold_standard
    #urls_docalign = args.urls_docalign
    #transient_prefix = args.transient_prefix

    if not os.path.isdir(prefix):
        raise Exception(f"Provided prefix has to be a directory: {prefix}")

    gold_standard_provided = True if gold_standard else False
    urls = []
    html = []
    text = []
    gs_urls = set()

    if gold_standard_provided:
        for gs_url in gold_standard:
            url = gs_url.rstrip('\n')

            gs_urls.add(url)

    if not gold_standard_provided or len(gs_urls) != 0:
        for idx, line in enumerate(data, 1):
            _lang, url, _html, _text = line.rstrip('\n').split('\t')

            if lang == _lang:
                urls.append(url)
                html.append(_html)
                text.append(_text)
            else:
                logging.debug("Line %d: lang different from src and trg: %s", idx, lang)

    with gzip.open(f"{prefix}/mime.gz", "wb") as mime_file, \
         gzip.open(f"{prefix}/url.gz",  "wb") as urls_file, \
         gzip.open(f"{prefix}/html.gz", "wb") as html_file, \
         gzip.open(f"{prefix}/text.gz", "wb") as text_file \
        :
        entries = 0

        for idx in range(len(urls)):
            gs_write = gold_standard_provided and urls[idx] in gs_urls
            write_data = not gold_standard_provided or gs_write

            if write_data:
                url = urls[idx]

                mime_file.write(f"{mime}\n".encode("utf-8", errors="ignore"))
                urls_file.write(f"{url}\n".encode("utf-8", errors="ignore"))
                html_file.write(f"{html[idx]}\n".encode("utf-8", errors="ignore"))
                text_file.write(f"{text[idx]}\n".encode("utf-8", errors="ignore"))

                entries += 1

        logging.info("Total entries: %d", entries)

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Create Bitextor preprocess files")

    parser.add_argument('data', type=argparse.FileType('rt'), help="Format: lang <tab> URL <tab> base64_html <tab> base64_text")
    parser.add_argument('prefix', help="Src prefix where src output files will be stored")
    #parser.add_argument('transient_prefix', help="Prefix where transient output files will be stored")

    parser.add_argument('--lang', required=True, help="Lang")
    parser.add_argument('--mime', default="text/plain", help="MIME value")
    parser.add_argument('--gold-standard', type=argparse.FileType('rt'), help="Gold standard file. If provided, only URLs which are in the GS, it will be processed")
    #parser.add_argument('--urls-docalign', type=argparse.FileType('rt'), help="URLs docalign file. If provided, the docalign index of matches will be created. Format: score <tab> src URL <tab> trg URL")

    parser.add_argument('-v', '--verbose', action="store_true", help="Verbose logging mode")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = initialization()

    utils.set_up_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    logging.debug("Arguments processed: {}".format(str(args)))

    main(args)
