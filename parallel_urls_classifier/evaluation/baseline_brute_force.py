
# Based on https://aclanthology.org/W16-2366.pdf 4.2

import os
import sys
import logging
import argparse

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, f"{cdir}/../..")

import parallel_urls_classifier.utils.utils as utils
import parallel_urls_classifier.tokenizer as tokenizer

import sklearn.metrics

def get_gs(file, lowercase=False):
    gs, src_gs, trg_gs = set(), set(), set()

    for idx, line in enumerate(file, 1):
        line = line.rstrip('\n').split('\t')

        if len(line) != 2:
            logging.warning("GS: unexpected number of fields in TSV entry #%d: %d were expected but got %d", idx, 2, len(line))

            continue

        if lowercase:
            line = [l.lower() for l in line]

        src_gs.add(line[0])
        trg_gs.add(line[1])
        gs.add('\t'.join(line))

    return gs, src_gs, trg_gs

def evaluate_pairs_m_x_n(src_lang, trg_lang, src_urls, trg_urls, use_gs, gs, src_gs, trg_gs, print_pairs=True):
    def evaluate(src_url, trg_url):
        src_url_tokenized = tokenizer.tokenize(src_url, check_gaps=False)
        trg_url_tokenized = tokenizer.tokenize(trg_url, check_gaps=False)
        parallel = 0

        # Replace and check
        normalized_url = [trg_lang if token == src_lang else token for token in src_url_tokenized]
        parallel = 1 if normalized_url == trg_url_tokenized else 0
        pair = f"{src_url}\t{trg_url}"

        if print_pairs and parallel:
            print(pair)

        y_pred = parallel
        y_true = 1 if pair in gs else 0

        return y_pred, y_true

    y_pred, y_true = [], []
    matches = 0
    total_pairs = 0

    for src_url in src_urls:
        remove_trg_urls_idx = None

        for idx, trg_url in enumerate(trg_urls):
            pair = False

            if use_gs:
                if src_url in src_gs or trg_url in trg_gs:
                    # Only append those URLs which are in the GS (we don't need to evaluate ALL the src and trg product URLs)
                    pair = True
            else:
                pair = True

            if pair:
                # Evaluate URL pair
                _y_pred, _y_true = evaluate(src_url, trg_url)

                y_pred.append(_y_pred)
                y_true.append(_y_true)

                total_pairs += 1

                if _y_pred:
                    remove_trg_urls_idx = idx

                    matches += 1

                    break # src_url now has a match, so we don't look for any other

        if remove_trg_urls_idx is not None:
            # Remove trg URLs which were identified as parallel (we don't need to do the same for the src URLs since they will
            #  be checked only once)
            del trg_urls[remove_trg_urls_idx]

    return y_pred, y_true, matches, total_pairs

def evaluate_section_42(src_lang, trg_lang, src_urls, trg_urls, use_gs, gs, src_gs, print_pairs=True):
    trg_urls_tokenized = [tokenizer.tokenize(trg_url, check_gaps=False) for trg_url in trg_urls]

    def evaluate(src_url):
        src_url_tokenized = tokenizer.tokenize(src_url, check_gaps=False)
        #trg_url_tokenized = tokenizer.tokenize(trg_url, check_gaps=False)
        parallel = 0

        # Replace and check
        normalized_url = [trg_lang if token == src_lang else token for token in src_url_tokenized]
        y_pred, y_true, trg_url_idx = None, None, None
        parallel = 0
        pair = ''

        try:
            trg_url_idx = trg_urls_tokenized.index(normalized_url)
            parallel = 1
            pair = f"{src_url}\t{trg_urls[trg_url_idx]}"
        except ValueError:
            # Normalized URL not found: is not a "possible pair"

            return y_pred, y_true, trg_url_idx

        if print_pairs and parallel:
            print(pair)

        y_pred = parallel
        y_true = 1 if pair in gs else 0

        return y_pred, y_true, trg_url_idx

    y_pred, y_true = [], []
    matches = 0
    total_pairs = 0

    for src_url in src_urls:
        remove_trg_urls_idx = None
        pair = False

        if use_gs:
            if src_url in src_gs:
                # Only append those URLs which are in the GS
                pair = True
        else:
            pair = True

        if pair:
            # Evaluate URL pair
            _y_pred, _y_true, remove_trg_urls_idx = evaluate(src_url)

            if remove_trg_urls_idx is not None:
                # We have a "possible pair"

                total_pairs += 1

                if _y_pred:
                    # Remove trg URLs which were identified as parallel (we don't need to do the same for the src URLs
                    #  since they will be checked only once)
                    del trg_urls[remove_trg_urls_idx]
                    del trg_urls_tokenized[remove_trg_urls_idx]

                    matches += 1

            y_pred.append(0 if _y_pred is None else _y_pred)
            y_true.append(1 if _y_true is None else _y_true) # The FN is forced when GS is not provided, but when has been provided,
                                                             #   this point is reached because src_url is in the GS, so is a genuine FN

    return y_pred, y_true, matches, total_pairs

def main(args):
    input_file = args.input
    src_lang = args.src_lang
    trg_lang = args.trg_lang
    gs_file = args.gold_standard
    evaluate_urls_in_gs = args.evaluate_urls_in_gs
    lowercase = args.lowercase
    evaluate_m_x_n = args.evaluate_m_x_n

    gs, src_gs, trg_gs = get_gs(gs_file, lowercase=lowercase) if gs_file else (set(), set(), set())
    src_urls, trg_urls = [], []
    use_gs = evaluate_urls_in_gs and gs_file

    # Read URLs
    for idx, line in enumerate(input_file, 1):
        line = line.rstrip('\n').split('\t')

        if len(line) != 2:
            logging.warning("Unexpected number of fields in TSV entry #%d: %d were expected but got %d", idx, 2, len(line))

            continue

        url, lang = line

        if lowercase:
            url = url.lower()

        if lang == src_lang:
            src_urls.append(url)
        elif lang == trg_lang:
            trg_urls.append(url)
        else:
            logging.warning("Unexpected lang in TSV entry #%d: %s", idx, lang)

    logging.info("Src and trg URLs: %d, %d", len(src_urls), len(trg_urls))

    # Create pairs of URLs to evaluate
    if use_gs:
        logging.debug("Only URLs which appears in the GS will be evaluated")
    else:
        logging.warning("GS will not be used for evaluation, so the product of all URLs will be added")

    logging.info("Evaluating...")

    # Evaluate
    if evaluate_m_x_n:
        y_pred, y_true, matches, total_pairs =\
            evaluate_pairs_m_x_n(src_lang, trg_lang, src_urls, trg_urls, use_gs, gs, src_gs, trg_gs)
    else:
        y_pred, y_true, matches, total_pairs =\
            evaluate_section_42(src_lang, trg_lang, src_urls, trg_urls, use_gs, gs, src_gs)

    # Some statistics
    negative_matches = total_pairs - matches

    logging.info("URL pairs: %d", total_pairs)
    logging.info("Positive and negative matches: %d, %d", matches, negative_matches)

    if gs_file:
        logging.info("Using GS in order to get some evaluation metrics...")

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
    parser.add_argument('--lowercase', action="store_true", help="Lowercase URLs (GS as well if provided). It might increase the evaluation results")
    parser.add_argument('--evaluate-m-x-n', action="store_true",
                        help="Evaluate all the possible pairs instead of construct 'possible pairs' like is described in section 4.2 of YODA system")

    parser.add_argument('-v', '--verbose', action="store_true", help="Verbose logging mode")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = initialization()

    utils.set_up_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    logging.debug("Arguments processed: {}".format(str(args)))

    main(args)
