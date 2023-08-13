
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

_get_same_case_warning_only_once = False
def get_same_case(s, reference, apply=True):
    if not apply:
        return s

    if len(reference) != len(s):
        global _get_same_case_warning_only_once

        if not _get_same_case_warning_only_once:
            logging.warning("Different lengths: can't apply the same case: returning without changes: "
                            "this message will be displayed only once")

            _get_same_case_warning_only_once = True

        return s

    result = ''

    for _s, _r in zip(s, reference):
        if _r.islower():
            result += _s.lower()
        elif _r.isupper():
            result += _s.upper()
        else:
            logging.warning("Reference '%s' (specifically: '%s') is not either lower or upper: returning without changes", reference, _r)

            return s

    return result

def evaluate_pairs_m_x_n(src_lang_tokens, trg_lang_tokens, src_urls, trg_urls, use_gs, gs, src_gs, trg_gs,
                         lowercase_tokens=False, print_pairs=True, print_negative_matches=False, print_score=False,
                         evaluate_pairs=False):
    def evaluate(src_url, trg_url):
        src_url_tokenized = tokenizer.tokenize(src_url, check_gaps=False)
        trg_url_tokenized = tokenizer.tokenize(trg_url, check_gaps=False)
        parallel = 0

        # Replace and check
        normalized_urls = [
            [get_same_case(trg_lang_token, token, apply=lowercase_tokens) if \
                (token.lower() if lowercase_tokens else token) == src_lang_token else token for token in src_url_tokenized]
            for src_lang_token, trg_lang_token in itertools.product(src_lang_tokens, trg_lang_tokens)
        ]
        found = 0
        pair = f"{src_url}\t{trg_url}"

        for normalized_url in normalized_urls:
            parallel = 1 if normalized_url == trg_url_tokenized else 0

            if parallel:
                found += 1

        if found > 0:
            parallel = 1

            if found > 1:
                logging.warning("More than 1 normalized URL found: %d: why? It shouldn't affect the result", found)

        if print_pairs and (parallel or print_negative_matches):
            if print_score:
                print(f"{pair}\t{parallel}")
            else:
                print(pair)

        y_pred = parallel
        y_true = 1 if pair in gs else 0

        return y_pred, y_true

    y_pred, y_true = [], []
    matches = 0
    total_pairs = 0
    seen_src_urls, seen_trg_urls = set(), set()

    for src_url_idx in range(len(src_urls)):
        for trg_url_idx in range(len(trg_urls)):
            if evaluate_pairs:
              trg_url_idx = src_url_idx

            src_url = src_urls[src_url_idx]
            trg_url = trg_urls[trg_url_idx]
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

                if src_url in seen_src_urls or trg_url in seen_trg_urls:
                    # Already evaluated pairs are "removed"
                    _y_pred = 0

                y_pred.append(_y_pred)
                y_true.append(_y_true)

                total_pairs += 1

                if _y_pred:
                    matches += 1

                    seen_src_urls.add(src_url)
                    seen_trg_urls.add(trg_url)

                    # We don't break because we need to evaluate all the URLs from the cart. product

            if evaluate_pairs:
                break

    return y_pred, y_true, matches, total_pairs

# TODO is evaluate_section_42 equivalent to evaluate_pairs_m_x_n?
def evaluate_section_42(src_lang_tokens, trg_lang_tokens, src_urls, trg_urls, use_gs, gs, src_gs,
                        lowercase_tokens=False, print_pairs=True, print_negative_matches=False, print_score=False):
    trg_urls_tokenized = ['\t'.join(tokenizer.tokenize(trg_url, check_gaps=False)) for trg_url in trg_urls] # Separate tokens using \t
    trg_urls_tokenized_set = set(trg_urls_tokenized)

    def evaluate(src_url):
        src_url_tokenized = tokenizer.tokenize(src_url, check_gaps=False)
        #trg_url_tokenized = tokenizer.tokenize(trg_url, check_gaps=False)
        parallel = 0

        # Replace and check
        normalized_urls = [
            # Separate tokens using \t
            '\t'.join([get_same_case(trg_lang_token, token, apply=lowercase_tokens) if \
                (token.lower() if lowercase_tokens else token) == src_lang_token else token for token in src_url_tokenized])
            for src_lang_token, trg_lang_token in itertools.product(src_lang_tokens, trg_lang_tokens)
        ]
        y_pred, y_true, trg_url_idx = None, None, None
        parallel = 0
        pair = ''
        found = 0

        for normalized_url in normalized_urls:
            if normalized_url not in trg_urls_tokenized_set:
                # Normalized URL not found: is not a "possible pair"

                continue

            trg_url_idx = trg_urls_tokenized.index(normalized_url)
            parallel = 1
            pair = f"{src_url}\t{trg_urls[trg_url_idx]}"
            found += 1

        if not found:
            # We don't have a trg_url, so we can't do anything else -> return

            return y_pred, y_true, trg_url_idx
        elif found > 1:
            logging.warning("More than 1 normalized URL found: %d: returning last match: it might affect the result", found)

        if print_pairs and parallel:
            if print_score:
                print(f"{pair}\t{parallel}") # Be aware that parallel will be always 1
            else:
                print(pair)

        y_pred = parallel
        y_true = 1 if pair in gs else 0

        return y_pred, y_true, trg_url_idx

    y_pred, y_true = [], []
    matches = 0
    total_pairs = 0
    seen_src_urls, seen_trg_urls = set(), set()
    possible_pairs = set()

    for src_url in src_urls:
        pair = False

        if use_gs:
            if src_url in src_gs:
                # Only append those URLs which are in the GS
                pair = True
        else:
            pair = True

        if pair:
            # Evaluate URL pair
            _y_pred, _y_true, trg_urls_idx = evaluate(src_url)

            if trg_urls_idx is not None:
                # We have a "possible pair"

                total_pairs += 1
                trg_url = trg_urls[trg_urls_idx]
                possible_pair = f"{src_url}\t{trg_url}"

                if src_url in seen_src_urls or trg_url in seen_trg_urls:
                    # Already evaluated pairs are "removed"
                    _y_pred = 0

                possible_pairs.add(possible_pair)

                if _y_pred:
                    seen_src_urls.add(src_url)
                    seen_trg_urls.add(trg_url)

                    matches += 1

                y_pred.append(_y_pred)
                y_true.append(_y_true)

        for trg_url in trg_urls:
            # Evaluate the rest of pairs: we can do it now since if a pair was found with src_url, it's been already detected
            pair = f"{src_url}\t{trg_url}"

            if pair in possible_pairs:
                # Already evaluated
                continue

            total_pairs += 1
            parallel = 0

            if print_pairs and (parallel or print_negative_matches):
                if print_score:
                    print(f"{pair}\t{parallel}")
                else:
                    print(pair)

            y_pred.append(parallel)
            y_true.append(1 if pair in gs else 0)

    return y_pred, y_true, matches, total_pairs

def main(args):
    input_file = args.input
    src_lang = args.src_lang
    trg_lang = args.trg_lang
    src_lang_tokens = args.src_lang_tokens
    trg_lang_tokens = args.trg_lang_tokens
    gs_file = args.gold_standard
    evaluate_urls_in_gs = args.evaluate_urls_in_gs
    lowercase_tokens = args.lowercase_tokens
    evaluate_m_x_n = args.evaluate_m_x_n
    print_negative_matches = args.print_negative_matches
    print_score = args.print_score
    input_are_pairs = args.input_are_pairs

    gs, src_gs, trg_gs = get_gs(gs_file) if gs_file else (set(), set(), set())
    src_urls, trg_urls = [], []
    use_gs = evaluate_urls_in_gs and gs_file

    if input_are_pairs and not evaluate_m_x_n:
      raise Exception("Not supported")

    # Read URLs
    for idx, line in enumerate(input_file, 1):
        line = line.rstrip('\n').split('\t')

        if len(line) != 2:
            logging.warning("Unexpected number of fields in TSV entry #%d: %d were expected but got %d", idx, 2, len(line))

            continue

        if input_are_pairs:
          src_url, trg_url = line

          src_urls.append(src_url)
          trg_urls.append(trg_url)
        else:
          url, lang = line

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
            evaluate_pairs_m_x_n(src_lang_tokens, trg_lang_tokens, src_urls, trg_urls, use_gs, gs, src_gs, trg_gs,
                                 lowercase_tokens=lowercase_tokens, print_negative_matches=print_negative_matches,
                                 print_score=print_score, evaluate_pairs=input_are_pairs)
    else:
        y_pred, y_true, matches, total_pairs =\
            evaluate_section_42(src_lang_tokens, trg_lang_tokens, src_urls, trg_urls, use_gs, gs, src_gs,
                                lowercase_tokens=lowercase_tokens, print_negative_matches=print_negative_matches,
                                print_score=print_score)

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

    parser.add_argument('input', type=argparse.FileType('rt', errors="backslashreplace"), help="Filename with URLs (TSV format). Format: URL <tab> lang")

    parser.add_argument('--src-lang', default="en", help="Src lang for the provided URL in the 1st column of the input file")
    parser.add_argument('--trg-lang', default="fr", help="Trg lang for the provided URL in the 1st column of the input file")
    parser.add_argument('--src-lang-tokens', nargs='+', default=["en"], # Found in WMT16 train: ["en", "eng", "e"]
                        help="Src tokens which will be used for checking the normalized URL (cart. product with --trg-lang-tokens)")
    parser.add_argument('--trg-lang-tokens', nargs='+', default=["fr"], # Found in WMT16 train: ["fr", "francais", "fra", "f", "fre", "french"]
                        help="Trg tokens which will be used for replacing in the normalized URL (cart. product with --src-lang-tokens)")
    parser.add_argument('--gold-standard', type=argparse.FileType('rt', errors="backslashreplace"), help="GS filename with parallel URLs (TSV format)")
    parser.add_argument('--evaluate-urls-in-gs', action="store_true",
                        help="Only evaluate those URLs which are present in the GS, not all the combinations of the provided input")
    parser.add_argument('--lowercase-tokens', action="store_true", help="Lowercase URL tokens (GS as well if provided). It might increase the evaluation results")
    parser.add_argument('--evaluate-m-x-n', action="store_true",
                        help="Evaluate all the possible pairs instead of construct 'possible pairs' like is described in section 4.2 of YODA system")
    parser.add_argument('--print-negative-matches', action="store_true",
                        help="Print negative matches (i.e. not only possitive matches)")
    parser.add_argument('--print-score', action="store_true",
                        help="Print 0 or 1 for positive or negative matches, respectively")
    parser.add_argument('--input-are-pairs', action="store_true",
                        help="Input is expected to be pairs instead of 'URL <tab> lang'. Input is expected to be 'URL1 <tab> URL2' where "
                             "URL1 is expected to be in language provided in --src-lang (same for URL2 and --trg-lang)")

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
