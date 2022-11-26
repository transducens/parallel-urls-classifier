
# Based on https://aclanthology.org/W16-2366.pdf 4.2

import os
import sys
import logging
import argparse
import itertools

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, f"{cdir}/../..")

import parallel_urls_classifier.utils.utils as utils
import parallel_urls_classifier.tokenizer as tokenizer

import sklearn.metrics

def get_gs(file):
    gs, src_gs, trg_gs = set(), set(), set()

    for idx, line in enumerate(file, 1):
        line = line.rstrip('\n').split('\t')

        if len(line) != 2:
            logging.warning("GS: unexpected number of fields in TSV entry #%d: %d were expected but got %d", idx, 2, len(line))

            continue

        src_gs.add(line[0])
        trg_gs.add(line[1])
        gs.add('\t'.join(line))

    return gs, src_gs, trg_gs

def evaluate_pair(src_lang, trg_lang, src_url, trg_url, gs, lowercase_tokens, print_pair=True):
    src_url_tokenized = tokenizer.tokenize(src_url.lower() if lowercase_tokens else src_url, check_gaps=False)
    trg_url_tokenized = tokenizer.tokenize(trg_url.lower() if lowercase_tokens else trg_url, check_gaps=False)
    parallel = 0

    # Replace and check
    normalized_url = [trg_lang if token == src_lang else token for token in src_url_tokenized]
    parallel = 1 if normalized_url == trg_url_tokenized else 0
    pair = f"{src_url}\t{trg_url}"

    if print_pair and parallel:
        print(pair)

    y_pred = parallel
    y_true = 1 if pair in gs else 0

    return y_pred, y_true

def main(args):
    input_file = args.input
    src_lang = args.src_lang
    trg_lang = args.trg_lang
    gs_file = args.gold_standard
    evaluate_urls_in_gs = args.evaluate_urls_in_gs
    lowercase_tokens = args.lowercase_tokens

    gs, src_gs, trg_gs = get_gs(gs_file) if gs_file else (set(), set(), set())
    y_true, y_pred = [], []
    src_urls, trg_urls = [], []
    total_pairs = 0

    # Read URLs
    for idx, line in enumerate(input_file, 1):
        line = line.rstrip('\n').split('\t')

        if len(line) != 2:
            logging.warning("Unexpected number of fields in TSV entry #%d: %d were expected but got %d", idx, 2, len(line))

            continue

        url, lang = line

        if lang == src_lang:
            src_urls.append(url)
        elif lang == trg_lang:
            trg_urls.append(url)
        else:
            logging.warning("Unexpected lang in TSV entry #%d: %s", idx, lang)

    # Create pairs of URLs to evaluate
    if evaluate_urls_in_gs and gs_file:
        logging.debug("Only URLs which appears in the GS will be evaluated")
    else:
        logging.warning("GS will not be used for evaluation, so the product of all URLs will be added")

    for src_url, trg_url in itertools.product(src_urls, trg_urls):
        pair = False

        if evaluate_urls_in_gs and gs_file:
            if src_url in src_gs or trg_url in trg_gs:
                # Only append those URLs which are in the GS (we don't need to evaluate ALL the src and trg product URLs)
                pair = True
        else:
            pair = True

        if pair:
            # Evaluate URL pair
            _y_pred, _y_true = evaluate_pair(src_lang, trg_lang, src_url, trg_url, gs, lowercase_tokens)

            y_pred.append(_y_pred)
            y_true.append(_y_true)

            total_pairs += 1

    logging.info("URL pairs: %d", total_pairs)
    logging.info("Evaluating...")

    if gs_file:
        # Log metrics
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = confusion_matrix.ravel()

        logging.info("GS: confusion matrix ([[negative and positive samples of class c] for c in classes]): %s", list(confusion_matrix))
        logging.info("GS: tn, fp, fn, tp: %d, %d, %d, %d", tn, fp, fn, tp)

        precision = [sklearn.metrics.precision_score(y_true, y_pred, labels=[0, 1], pos_label=cls) for cls in [0, 1]]
        recall = [sklearn.metrics.recall_score(y_true, y_pred, labels=[0, 1], pos_label=cls) for cls in [0, 1]]
        f1 = [sklearn.metrics.f1_score(y_true, y_pred, labels=[0, 1], pos_label=cls) for cls in [0, 1]]

        logging.info("GS: precision: %s", precision)
        logging.info("GS: recall: %s", recall)
        logging.info("GS: F1: %s", f1)

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Classify URLs using brute force method")

    parser.add_argument('input', type=argparse.FileType('rt'), help="Filename with URLs (TSV format). Format: URL <tab> lang")

    parser.add_argument('--src-lang', default="en", help="Src lang for the provided URL in the 1st column of the input file")
    parser.add_argument('--trg-lang', default="fr", help="Trg lang for the provided URL in the 1st column of the input file")
    parser.add_argument('--gold-standard', type=argparse.FileType('rt'), help="GS filename with parallel URLs (TSV format)")
    parser.add_argument('--evaluate-urls-in-gs', action="store_true", help="Only evaluate those URLs which are present in the GS")
    parser.add_argument('--lowercase-tokens', action="store_true", help="Lowercase URL tokens. It might increase the evaluation results")

    parser.add_argument('-v', '--verbose', action="store_true", help="Verbose logging mode")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = initialization()

    utils.set_up_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    logging.debug("Arguments processed: {}".format(str(args)))

    main(args)
