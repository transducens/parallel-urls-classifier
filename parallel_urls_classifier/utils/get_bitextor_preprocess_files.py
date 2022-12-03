
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
    src_prefix = args.src_prefix
    trg_prefix = args.trg_prefix
    src_lang = args.src_lang
    trg_lang = args.trg_lang
    mime = args.mime
    gold_standard = args.gold_standard
    gold_standard_strict = args.gold_standard_strict
    urls_docalign = args.urls_docalign
    transient_prefix = args.transient_prefix

    if not os.path.isdir(src_prefix):
        raise Exception(f"Provided src prefix has to be a directory: {src_prefix}")
    if not os.path.isdir(trg_prefix):
        raise Exception(f"Provided trg prefix has to be a directory: {trg_prefix}")

    gold_standard_provided = True if gold_standard else False
    src_urls, trg_urls = [], []
    src_html, trg_html = [], []
    src_text, trg_text = [], []
    src_gs, trg_gs = set(), set()

    if gold_standard_provided:
        for gs in gold_standard:
            src, trg = gs.rstrip('\n').split('\t')

            src_gs.add(src)
            trg_gs.add(trg)

    if not gold_standard_provided or len(src_gs) != 0 or len(trg_gs) != 0:
        for idx, line in enumerate(data, 1):
            lang, url, html, text = line.rstrip('\n').split('\t')

            if lang == src_lang:
                urls_arr = src_urls
                html_arr = src_html
                text_arr = src_text
            elif lang == trg_lang:
                urls_arr = trg_urls
                html_arr = trg_html
                text_arr = trg_text
            else:
                logging.debug("Line %d: lang different from src and trg: %s", idx, lang)
                continue

            urls_arr.append(url)
            html_arr.append(html)
            text_arr.append(text)

    scores_docalign = {}
    #pairs_urls_docalign = set()

    if urls_docalign:
        for idx, pair_docalign in enumerate(urls_docalign):
            score, src_url, trg_url = pair_docalign.rstrip('\n').split('\t')
            pair_urls = f"{src_url}\t{trg_url}"

            if pair_urls in scores_docalign:
                logging.warning("Docalign pair %d duplicated: %s", idx + 1, pair_urls)

                continue

            scores_docalign[pair_urls] = score
            #pairs_urls_docalign.add(pair_urls)

    bool_function = _and if gold_standard_strict else _or
    create_docalign = len(scores_docalign) > 0
    docalign_file = f"{transient_prefix}/{src_lang}1_{trg_lang}1.bitextor.06_01.matches"

    with gzip.open(f"{src_prefix}/mime.gz", "wb") as src_mime_file, \
         gzip.open(f"{src_prefix}/url.gz",  "wb") as src_urls_file, \
         gzip.open(f"{src_prefix}/html.gz", "wb") as src_html_file, \
         gzip.open(f"{src_prefix}/text.gz", "wb") as src_text_file, \
         gzip.open(f"{trg_prefix}/mime.gz", "wb") as trg_mime_file, \
         gzip.open(f"{trg_prefix}/url.gz",  "wb") as trg_urls_file, \
         gzip.open(f"{trg_prefix}/html.gz", "wb") as trg_html_file, \
         gzip.open(f"{trg_prefix}/text.gz", "wb") as trg_text_file, \
         open(docalign_file, "wt") if create_docalign else nullcontext() as docalign_file \
        :
        idx = 0
        src_entries, trg_entries = 0, 0
        trg_idxs = set()
        index = {"src": {}, "trg": {}}

        for src_idx in range(len(src_urls)):
            src_added = False

            for trg_idx in range(len(trg_urls)):
                gs_src_write = gold_standard_provided and src_urls[src_idx] in src_gs
                gs_trg_write = gold_standard_provided and trg_urls[trg_idx] in trg_gs
                write_data = bool_function(not gold_standard_provided or gs_src_write, not gold_standard_provided or gs_trg_write)

                if write_data:
                    src_url = src_urls[src_idx]
                    trg_url = trg_urls[trg_idx]
                    pair = f"{src_url}\t{trg_url}"

                    if not src_added:
                        src_mime_file.write(f"{mime}\n".encode("utf-8", errors="backslashreplace"))
                        src_urls_file.write(f"{src_url}\n".encode("utf-8", errors="backslashreplace"))
                        src_html_file.write(f"{src_html[src_idx]}\n".encode("utf-8", errors="backslashreplace"))
                        src_text_file.write(f"{src_text[src_idx]}\n".encode("utf-8", errors="backslashreplace"))

                        src_entries += 1

                        src_added = True
                        index["src"][src_url] = src_entries # Start: 1

                    if trg_idx not in trg_idxs:
                        trg_mime_file.write(f"{mime}\n".encode("utf-8", errors="backslashreplace"))
                        trg_urls_file.write(f"{trg_url}\n".encode("utf-8", errors="backslashreplace"))
                        trg_html_file.write(f"{trg_html[trg_idx]}\n".encode("utf-8", errors="backslashreplace"))
                        trg_text_file.write(f"{trg_text[trg_idx]}\n".encode("utf-8", errors="backslashreplace"))

                        trg_idxs.add(trg_idx)

                        trg_entries += 1

                        index["trg"][trg_url] = trg_entries # Start: 1

                    if create_docalign and pair in scores_docalign:
                        # Find out docalign URLs index

                        docalign_file.write(f"{index['src'][src_url]}\t{index['trg'][trg_url]}\t{scores_docalign[pair]}\n")

        logging.info("Total (src, trg) entries: (%d, %d)", src_entries, trg_entries)

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Create Bitextor preprocess files")

    parser.add_argument('data', type=argparse.FileType('rt', errors="backslashreplace"), help="Format: lang <tab> URL <tab> base64_html <tab> base64_text")
    parser.add_argument('src_prefix', help="Src prefix where src output files will be stored")
    parser.add_argument('trg_prefix', help="Trg prefix where trg output files will be stored")
    parser.add_argument('transient_prefix', help="Prefix where transient output files will be stored")

    parser.add_argument('--src-lang', required=True, help="Src lang")
    parser.add_argument('--trg-lang', required=True, help="Trg lang")
    parser.add_argument('--mime', default="text/plain", help="MIME value")
    parser.add_argument('--gold-standard', type=argparse.FileType('rt', errors="backslashreplace"), help="Gold standard file. If provided, only URLs which are in the GS, it will be processed")
    parser.add_argument('--gold-standard-strict', action="store_true", help="Gold standard pairs will have to be strict matches instead of just one side to be present")
    parser.add_argument('--urls-docalign', type=argparse.FileType('rt', errors="backslashreplace"), help="URLs docalign file. If provided, the docalign index of matches will be created. Format: score <tab> src URL <tab> trg URL")

    parser.add_argument('-v', '--verbose', action="store_true", help="Verbose logging mode")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    # https://stackoverflow.com/questions/16549332/python-3-how-to-specify-stdin-encoding
    sys.stdin.reconfigure(encoding='utf-8', errors="backslashreplace")

    args = initialization()

    utils.set_up_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    logging.debug("Arguments processed: {}".format(str(args)))

    main(args)
