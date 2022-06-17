
import sys
import gzip
import logging
import argparse
from contextlib import nullcontext

import utils

def main(args):
    src_urls_data = args.src_urls_data
    trg_urls_data = args.trg_urls_data
    classifier_data = args.classifier_data
    transient_prefix = args.transient_prefix
    header_column_score = args.header_column_score

    index_urls = {"src": {}, "trg": {}}

    with gzip.open(src_urls_data, "rb") if src_urls_data != '-' else sys.stdin as urls_data:
        for idx, url in enumerate(urls_data, 1):
            url = url.decode("utf-8", errors="ignore").rstrip('\n')
            index_urls["src"][url] = idx

    with gzip.open(trg_urls_data, "rb") if trg_urls_data != '-' else sys.stdin as urls_data:
        for idx, url in enumerate(urls_data, 1):
            url = url.decode("utf-8", errors="ignore").rstrip('\n')
            index_urls["trg"][url] = idx

    logging.info("Total (src, trg) entries (deduplicated): (%d, %d)", len(index_urls["src"]), len(index_urls["trg"]))

    scores_docalign = {}

    for idx, pair_docalign in enumerate(classifier_data, 1):
        score, src_url, trg_url = pair_docalign.rstrip('\n').split('\t')
        pair_urls = f"{src_url}\t{trg_url}"

        if pair_urls in scores_docalign:
            logging.warning("Docalign pair %d duplicated: %s", idx, pair_urls)

            continue

        scores_docalign[pair_urls] = score

    #docalign_file = f"{transient_prefix}/{src_lang}1_{trg_lang}1.bitextor.06_01.matches"
    docalign_file = transient_prefix
    results = []

    with open(docalign_file, "wt") if transient_prefix != '-' else nullcontext(sys.stdout) as docalign_file:
        docalign_file.write(f"src_index\ttrg_index\t{header_column_score}\n")

        for pair in scores_docalign:
            src_url, trg_url = pair.split('\t')

            if src_url in index_urls["src"] and trg_url in index_urls["trg"]:
                results.append((index_urls["src"][src_url], index_urls["trg"][trg_url], scores_docalign[pair]))

        results.sort()

        for src_url_idx, trg_url_idx, score in results:
            docalign_file.write(f"{src_url_idx}\t{trg_url_idx}\t{score}\n")

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Create Bitextor docalign file")

    parser.add_argument('src_urls_data', help="Src URLs. Gzip file")
    parser.add_argument('trg_urls_data', help="Trg URLs. Gzip file")
    parser.add_argument('classifier_data', type=argparse.FileType('rt'), help="Format: score <tab> src_url <tab> trg_url")
    parser.add_argument('transient_prefix', help="Prefix where transient output files will be stored")

    parser.add_argument('--header-column-score', default="dic_doc_aligner_score", help="Header column name of the score")

    parser.add_argument('-v', '--verbose', action="store_true", help="Verbose logging mode")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = initialization()

    utils.set_up_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    logging.debug("Arguments processed: {}".format(str(args)))

    main(args)
