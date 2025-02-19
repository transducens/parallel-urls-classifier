
import os
import sys
import logging
import argparse

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, f"{cdir}/../..")

import parallel_urls_classifier.utils.utils as utils

import numpy as np

def main(args):
    urls_file = args.urls_file
    gold_standard_file = args.gold_standard_file

    parallel_urls, non_parallel_urls = 0, 0
    src_gs_urls, trg_gs_urls = [], []
    gs_pairs = set()
    tp, fp, fn, tn = 0, 0, 0, 0

    with utils.open_xz_or_gzip_or_plain(gold_standard_file) as f:
        for l in f:
            urls = l.strip().split('\t')

            if len(urls) != 2:
                raise Exception(f"Unexpected GS format: 2 vs {len(urls)}")

            src_gs_urls.append(urls[0])
            trg_gs_urls.append(urls[1])
            gs_pairs.add(f"{urls[0]}\t{urls[1]}")

    logging.info("GS pairs: %d", len(gs_pairs))

    with utils.open_xz_or_gzip_or_plain(urls_file) as f:
        for l in f:
            urls = l.strip().split('\t')

            if len(urls) != 3:
                raise Exception(f"Unexpected URLs format: 3 vs {len(urls)}")

            is_parallel = urls[0]
            src_url = urls[1]
            trg_url = urls[2]
            pair = f"{src_url}\t{trg_url}"

            if is_parallel not in ("parallel", "non-parallel"):
                raise Exception(f"Unexpected parallel value: {is_parallel}")

            if is_parallel == "parallel":
                parallel_urls += 1

                if pair in gs_pairs:
                    tp += 1
                else:
                    fp += 1
            else:
                non_parallel_urls += 1

                if pair in gs_pairs:
                    fn += 1
                else:
                    tn += 1

    logging.info("Provided parallel and non-parallel URL pairs: %d, %d", parallel_urls, non_parallel_urls)

    if tp + fn != len(gs_pairs):
        logging.warning("Not all pairs from GS were classified in the provided data, so metrics might not be trustworthy: "
                        "only %d of %d pairs were provided", tp + fn, len(gs_pairs))

    logging.info("tp, tn: %d, %d", tp, tn)
    logging.info("fp, fn: %d, %d", fp, fn)

    precision = tp / (tp + fp) if (tp + fp) != 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 1.0
    f1 = 2 * ((precision * recall) / (precision + recall)) if not np.isclose(precision + recall, 0.0) else 0.0

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Recall and precision evaluation")

    parser.add_argument('urls_file', help="Input file with the following format: 'parallel'|'non-parallel'<tab>src_url<tab>trg_url")
    parser.add_argument('gold_standard_file', help="Gold standard file")

    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose logging mode")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = initialization()

    utils.set_up_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    logging.debug("Arguments processed: {}".format(str(args)))

    main(args)