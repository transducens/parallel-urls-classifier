
import os
import sys
import base64
import logging
import argparse
import itertools
import subprocess

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, f"{cdir}/../..")

#import parallel_urls_classifier.utils.levenshtein as levenshtein
import parallel_urls_classifier.utils.utils as utils

import numpy as np
import Levenshtein

def process_pairs(pairs, command, results_fd=None, results_are_fp=False, do_not_classify_missing_pairs=True):
    def log_classifier_stderr(msg):
        logging.warning("There were errors, so FD/classifier output is going to be displayed")

        for idx, e in enumerate(msg):
            logging.warning("Stderr line %d: %s", idx, e)

    results = []
    error = False
    list_results = []
    classifier_output = []
    obtained_pairs = []

    if results_fd:
        clean_pairs = set(map(lambda p: p.encode("utf-8", errors="ignore").decode(), pairs))
        set_pairs = set(pairs)
        missing_pairs = 0
        seen_pairs = set()

        for idx, l in enumerate(results_fd):
            l = l.strip()

            classifier_output.append(f"from FD: {l}")

            l = l.split('\t')

            if len(l) != 3:
                raise Exception(f"Line {idx + 1} doesn't have the expected format: {len(l)} columns vs 3 columns")

            v, src_pair, trg_pair = l
            pair = f"{src_pair}\t{trg_pair}"
            pair_reversed = f"{trg_pair}\t{src_pair}"

            # Append only the provided pairs which are inside the pairs that we want to classify
            if pair in set_pairs or pair in clean_pairs:
                if pair not in seen_pairs:
                    list_results.append((v, src_pair, trg_pair))
                    obtained_pairs.append(pair)
                    seen_pairs.add(pair)
            elif pair_reversed in set_pairs or pair_reversed in clean_pairs:
                # Fix direction
                if pair_reversed not in seen_pairs:
                    list_results.append((v, trg_pair, src_pair))
                    obtained_pairs.append(pair_reversed)
                    seen_pairs.add(pair_reversed)
            else:
                # It has failed, but we still might have the pair and have skipped it due to encoding

                src_pair_2 = src_pair.encode(errors="ignore").decode("utf-8", errors="ignore")
                trg_pair_2 = trg_pair.encode(errors="ignore").decode("utf-8", errors="ignore")
                pair_2 = f"{src_pair_2}\t{trg_pair_2}"
                pair_reversed_2 = f"{trg_pair_2}\t{src_pair_2}"

                if pair_2 in set_pairs or pair_2 in clean_pairs:
                    if pair_2 not in seen_pairs:
                        list_results.append((v, src_pair_2, trg_pair_2))
                        obtained_pairs.append(pair_2)
                        seen_pairs.add(pair_2)
                elif pair_reversed_2 in set_pairs or pair_reversed_2 in clean_pairs:
                    # Fix direction
                    if pair_reversed_2 not in seen_pairs:
                        list_results.append((v, trg_pair_2, src_pair_2))
                        obtained_pairs.append(pair_reversed_2)
                        seen_pairs.add(pair_reversed_2)
                else:
                    missing_pairs += 1
                #    logging.error("Missing pair (1): %s", pair)
                #    logging.error("Missing pair (2): %s", pair_2)

        logging.debug("Missing pairs: %d", missing_pairs)

    if not results_fd or (not do_not_classify_missing_pairs and len(obtained_pairs) < len(pairs)):
        classifier_list_results = pairs

        if results_fd and len(obtained_pairs) != 0:
            classifier_list_results = [a for a in pairs if a not in obtained_pairs]

            logging.warning("Not all results were provided: %d elements will be calculated with the classifier: evaluation might change", len(classifier_list_results))

        sp_result = subprocess.Popen(f"{command}", shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        aux_result, aux_err = sp_result.communicate('\n'.join(classifier_list_results).encode(errors="ignore"))

        list_results.extend(list(map(lambda v: v.split('\t'), aux_result.decode("utf-8", errors="ignore").strip().split('\n'))))
        classifier_output.extend(list(map(lambda e: f"from classifier command (subprocess): {e}", aux_err.decode("utf-8", errors="ignore").strip().split('\n'))))

    if not do_not_classify_missing_pairs and len(list_results) != len(pairs):
        log_classifier_stderr(classifier_output)

        raise Exception(f"Pairs length != classifier results length: {len(pairs)} vs {len(list_results)}")

    if do_not_classify_missing_pairs:
        if results_fd and len(obtained_pairs) != 0:
            missing_pairs_classification_len = len([a for a in pairs if a not in obtained_pairs])

            logging.warning("Not all results were provided: %d elements: missing elements will NOT be calculated", missing_pairs_classification_len)

    for idx, (v, src_pair, trg_pair) in enumerate(list_results):
        if results_are_fp:
            try:
                results.append((float(v), src_pair, trg_pair))
            except ValueError as e:
                logging.error("%s: returning scores of 0.0 (idx %d)", str(e), idx)

                results.append((0.0, src_pair, trg_pair))

                error = True
        else:
            if v not in ("non-parallel", "parallel"):
                logging.error("Unexpected value from URL classifier: returning URL pair as non-parallel (idx %d): %s", idx, str(v))

                results.append((False, src_pair, trg_pair))
            else:
                results.append((v == "parallel", src_pair, trg_pair))

    if error:
        log_classifier_stderr(classifier_output)

    return results

def evaluate_recall(src_pairs, trg_pairs, src_gs_pairs, trg_gs_pairs, src_urls, trg_urls, src_docs, trg_docs,
                    rule_1_1=True, disable_near_matchs=False, non_src_pairs=None, non_trg_pairs=None,
                    src_pairs_scores=None, trg_pairs_scores=None):
    tp, fp = 0, 0
    seen_src_pairs, seen_trg_pairs = set(), set()
    gs_pairs = set(f"{src_gs_pair}\t{trg_gs_pair}" for src_gs_pair, trg_gs_pair in zip(src_gs_pairs, trg_gs_pairs))
    positive_near_matches, negative_near_matches = 0, 0

    for idx, (src_pair, trg_pair) in enumerate(zip(src_pairs, trg_pairs)):
        pair = f"{src_pair}\t{trg_pair}"
        pair_hit = False
        near_match = False

        #if pair in gs_pairs and src_pair not in seen_src_pairs and trg_pair not in seen_trg_pairs:
        if pair in gs_pairs:
            if rule_1_1:
                if src_pair not in seen_src_pairs and trg_pair not in seen_trg_pairs:
                    tp += 1
                    pair_hit = True
            else:
                tp += 1
                pair_hit = True
        elif not disable_near_matchs:
            # "Soft" recall

            if src_pair in src_gs_pairs and trg_pair in trg_gs_pairs:
                # The strict pair is not in the GS, but each URL is in there, so we got a FP
                pass
            else:
                # Near-matches
                near_match_src = src_pair in src_gs_pairs and trg_pair not in trg_gs_pairs
                near_match_trg = trg_pair in trg_gs_pairs and src_pair not in src_gs_pairs

                if rule_1_1:
                    near_match_src = near_match_src and src_pair not in seen_src_pairs and trg_pair not in seen_trg_pairs
                    near_match_trg = near_match_trg and trg_pair not in seen_trg_pairs and src_pair not in seen_src_pairs

                if near_match_src and near_match_trg:
                    logging.error("This should never happen")
                elif near_match_src or near_match_trg:
                    if near_match_src:
                        url_1 = trg_pair
                        url_2 = trg_gs_pairs[src_gs_pairs.index(src_pair)]
                        doc_1_idx = trg_urls.index(url_1)
                        doc_2_idx = trg_urls.index(url_2)
                        doc_1 = trg_docs[doc_1_idx]
                        doc_2 = trg_docs[doc_2_idx]
                        src_gs_pair = src_pair
                        trg_gs_pair = url_2
                    else:
                        url_1 = src_pair
                        url_2 = src_gs_pairs[trg_gs_pairs.index(trg_pair)]
                        doc_1_idx = src_urls.index(url_1)
                        doc_2_idx = src_urls.index(url_2)
                        doc_1 = src_docs[doc_1_idx]
                        doc_2 = src_docs[doc_2_idx]
                        src_gs_pair = url_2
                        trg_gs_pair = trg_pair

                    logging.debug("Near-match?\t%s\t%s", url_1, url_2)
                    logging.debug("(GS, Not GS) pair:\t%s\t%s\t%s\t%s", src_gs_pair, trg_gs_pair, src_pair, trg_pair)

                    nolines_doc_1 = doc_1.strip().count('\n') + (1 if doc_1.strip() != '' else 0)
                    nolines_doc_2 = doc_2.strip().count('\n') + (1 if doc_2.strip() != '' else 0)
                    early_stopping = abs(nolines_doc_1 - nolines_doc_2) * 75 if max(nolines_doc_1, nolines_doc_2) > 10 else None
                    lev_distance = Levenshtein.distance(doc_1, doc_2, score_cutoff=early_stopping if early_stopping != 0 else None)

                    if early_stopping and lev_distance == early_stopping + 1:
                        # Early stopping hit
                        similarity = 0.0
                    else:
                        # Calculate actual similarity
                        similarity = 1.0 - lev_distance / max(len(doc_1), len(doc_2))

                    #early_stopping = abs(nolines_doc_1 - nolines_doc_2) * 75.0 if max(nolines_doc_1, nolines_doc_2) > 10 else np.inf
                    #similarity = levenshtein.levenshtein_opt_space_and_band(doc_1, doc_2, nfactor=max(len(doc_1), len(doc_2)), percentage=0.06, early_stopping=early_stopping)["similarity"]

                    logging.debug("Near-match similarity (url_1, url_2, similarity_score):\t%s\t%s\t%f", url_1, url_2, similarity)

                    if similarity >= 0.95:
                        logging.debug("Near-match found")

                        tp += 1
                        positive_near_matches += 1
                        pair_hit = True
                        near_match = True
                    else:
                        negative_near_matches += 1

        nm_mark = "[NM]" if near_match else ''
        tp_mark = "[TP]" if pair_hit else "[FP]"

        if src_pairs_scores and trg_pairs_scores:
            src_pairs_score = src_pairs_scores[idx]
            trg_pairs_score = trg_pairs_scores[idx]

            logging.debug("Pair (%s%s): %s\t%s\t%s\t%s", nm_mark, tp_mark, src_pair, trg_pair, src_pairs_score, trg_pairs_score)
        else:
            logging.debug("Pair (%s%s): %s\t%s", nm_mark, tp_mark, src_pair, trg_pair)

        if not pair_hit:
            fp += 1

        seen_src_pairs.add(src_pair)
        seen_trg_pairs.add(trg_pair)

    if not disable_near_matchs:
        logging.debug("(Positive, Negative) near-matches found: (%d, %d)", positive_near_matches, negative_near_matches)

    logging.info("(True, False) positives: (%d, %d)", tp, fp)

    tn, fn = 0, 0

    if non_src_pairs and non_trg_pairs:
        # Calculate TN and FN

        for non_src_pair, non_trg_pair in zip(non_src_pairs, non_trg_pairs):
            pair = f"{non_src_pair}\t{non_trg_pair}"

            if pair in gs_pairs:
                fn += 1
            else:
                tn += 1
    else:
        logging.debug("TN and FN could not be calculated")

    logging.info("(True, False) negatives: (%d, %d)", tn, fn)
    logging.info("GS pairs: %d", len(gs_pairs))
    logging.debug("GS is not exhaustive, so we cannot trust false positives, so we cannot trust precision")

    if len(gs_pairs) == 0:
        logging.warning("GS does not contain values")

    parallel_pairs_found = tp + fn
    expected_pairs_found = len(gs_pairs)

    if rule_1_1:
        # We need to add PNM and subtract NNM because NM will be classified as TP or FP, and the real GS pair (it should be
        #  among the pairs) will be classified as FP if rule 1-1 is enabled
        expected_pairs_found += 2 * positive_near_matches - negative_near_matches

    if parallel_pairs_found != expected_pairs_found:
        logging.error("Unexpected GS pairs found: %d were expected, %d were found", expected_pairs_found, parallel_pairs_found)

    recall = tp / len(gs_pairs) if len(gs_pairs) != 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 1.0

    print(f"Recall: {recall}")
    print(f"Precision (not trustworthy because GS is not exhaustive): {precision}")

def main(args):
    input_file = args.input_file
    gold_standard_file = args.gold_standard_file
    classifier_command = args.classifier_command
    classifier_results = args.classifier_results
    results_are_fp = args.results_are_fp
    parallel_threshold = args.parallel_threshold
    rule_1_1 = not args.disable_rule_1_1
    disable_near_matchs = args.disable_near_matchs
    do_not_classify_missing_pairs = args.do_not_classify_missing_pairs

    src_urls, trg_urls = [], []
    src_docs, trg_docs = [], []
    parallel = []
    domain = None

    for line in input_file:
        lang, url, doc = line.rstrip('\n').split('\t')
        d = url[8:] if url[:8] == "https://" else url[7:] if url[:7] == "http://" else url
        d = d.split('/')[0]

        if domain is None:
            domain = d
        if d != domain:
            raise Exception(f"Provided different domain: '{d}' vs '{domain}'")

        if lang not in ("en", "fr"):
            raise Exception(f"Unexpected lang (expected: en, fr): {lang}")

        doc = base64.b64decode(doc).decode("utf-8", errors="ignore").strip()

        if lang == "en":
            src_urls.append(url)
            src_docs.append(doc)
        else:
            trg_urls.append(url)
            trg_docs.append(doc)

    src_gs, trg_gs = [], []
    gs_entries = 0

    for line in gold_standard_file:
        src, trg = line.strip().split('\t')
        src_domain = src[8:] if src[:8] == "https://" else src[7:] if src[:7] == "http://" else src
        trg_domain = trg[8:] if trg[:8] == "https://" else trg[7:] if trg[:7] == "http://" else trg
        src_domain = src_domain.split('/')[0]
        trg_domain = trg_domain.split('/')[0]

        if src_domain != trg_domain:
            logging.warning(f"Different GS src and trg domain: '{src_domain}' vs '{trg_domain}'")

        if domain == src_domain and domain == trg_domain:
            src_gs.append(src)
            trg_gs.append(trg)

        gs_entries += 1

    logging.info("Provided entries (src, trg): (%d, %d)", len(src_urls), len(trg_urls))
    logging.info("GS entries: %d from %d", len(src_gs), gs_entries)

    pairs = []

    logging.info("Classifying...")

    # Prepare pairs in order to classify them
    for src_url, trg_url in itertools.product(src_urls, trg_urls):
        if src_url in src_gs or trg_url in trg_gs:
            # Only append those URLs which are in the GS (we don't need to evaluate ALL the src and trg product URLs)
            pairs.append((src_url, trg_url))

    logging.info("Pairs to be classified: %d", len(pairs))

    #time.sleep(10) # Sleep in order to try to avoid CUDA error out of memory

    if len(pairs) != 0:
        # We need to provide a list as first argument since rule 1-1 might produce different results every execution if we use a set
        formatted_pairs = list(map(lambda p: '\t'.join(p), pairs))
        parallel = process_pairs(formatted_pairs, classifier_command, results_fd=classifier_results, results_are_fp=results_are_fp,
                                 do_not_classify_missing_pairs=do_not_classify_missing_pairs)

    expected_values = len(src_gs) * len(trg_urls) + len(trg_gs) * len(src_urls) - len(src_gs) * len(trg_gs)

    if do_not_classify_missing_pairs:
        if expected_values != len(parallel):
            logging.warning("Unexpected parallel length: %d vs %d", expected_values, len(parallel))
        if len(pairs) != len(parallel):
            logging.warning("Unexpected parallel length: %d vs %d", len(pairs), len(parallel))
    else:
        assert expected_values == len(parallel), f"Unexpected parallel length: {expected_values} vs {len(parallel)}"
        assert len(pairs) == len(parallel), f"Unexpected parallel length: {len(pairs)} vs {len(parallel)}"

    # Update pairs in case that the order changed
    if len(parallel) == 0:
        parallel_classification, aux_src_pairs, aux_trg_pairs = (), (), ()
    else:
        parallel_classification, aux_src_pairs, aux_trg_pairs = zip(*parallel)

    pairs = list(zip(aux_src_pairs, aux_trg_pairs))

    if results_are_fp:
        parallel_values = sum(i >= parallel_threshold for i in parallel_classification)
        non_parallel_values = sum(i < parallel_threshold for i in parallel_classification)
    else:
        parallel_values = parallel_classification.count(True)
        non_parallel_values = parallel_classification.count(False)

    assert parallel_values + non_parallel_values == len(parallel), f"Unexpected parallel and non-parallel values: {parallel_values + non_parallel_values} vs {len(parallel)}"

    logging.info(f"(parallel, non-parallel): (%d, %d)", parallel_values, non_parallel_values)

    for v, (src_url, trg_url) in zip(parallel_classification, pairs):
        v = v if results_are_fp else ('parallel' if v else 'non-parallel')

        logging.debug(f"{v}\t{src_url}\t{trg_url}")

    src_pairs, trg_pairs = [], []
    src_pairs_scores, trg_pairs_scores = [], []
    non_src_pairs, non_trg_pairs = [], []

    for p, (src_url, trg_url) in zip(parallel_classification, pairs):
        if results_are_fp and p >= parallel_threshold:
            src_pairs.append((p, src_url))
            trg_pairs.append((p, trg_url))
        elif not results_are_fp and p:
            src_pairs.append(src_url)
            trg_pairs.append(trg_url)
        else:
            non_src_pairs.append(src_url)
            non_trg_pairs.append(trg_url)

    if results_are_fp:
        logging.debug("Sorting by score")

        if len(src_pairs) != 0 and len(trg_pairs) != 0:
            # Sort by score
            src_pairs, src_pairs_scores = zip(*list(map(lambda t: (t[1], t[0]), sorted(src_pairs, key=lambda v: v[0], reverse=True))))
            trg_pairs, trg_pairs_scores = zip(*list(map(lambda t: (t[1], t[0]), sorted(trg_pairs, key=lambda v: v[0], reverse=True))))
    else:
        src_pairs_scores = ["parallel"] * len(src_pairs)
        trg_pairs_scores = ["parallel"] * len(trg_pairs)

    logging.info("Parallel pairs: %d", len(src_pairs))

    assert len(src_pairs) + len(non_src_pairs) == len(pairs), f"Unexpected parallel and non-parallel length compared to pairs: {len(src_pairs)} + {len(non_src_pairs)} vs {len(pairs)}"
    assert len(src_pairs) == parallel_values, f"Unexpected quantity of parallel values: {len(src_pairs)} vs {parallel_values}"
    assert len(non_src_pairs) == non_parallel_values, f"Unexpected quantity of non-parallel values: {len(non_src_pairs)} vs {non_parallel_values}"

    evaluate_recall(src_pairs, trg_pairs, src_gs, trg_gs, src_urls, trg_urls, src_docs, trg_docs,
                    rule_1_1=rule_1_1, disable_near_matchs=disable_near_matchs,
                    non_src_pairs=non_src_pairs, non_trg_pairs=non_trg_pairs,
                    src_pairs_scores=src_pairs_scores, trg_pairs_scores=trg_pairs_scores)

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="WMT16 evaluation for a single domain")

    parser.add_argument('input_file', type=argparse.FileType('rt', errors="ignore"), help="Input file from the test set with the following format: lang<tab>URL<tab>base64-doc (they should be 1st, 4th and 6th column in the original WMT16 test set)")
    parser.add_argument('gold_standard_file', type=argparse.FileType('rt'), help="Gold standard file")

    parser.add_argument('--classifier-command', required=True, help="Classifier command whose expected output format is: class<tab>src_url<tab>trg_url (class is expected to be 'parallel'/'non-parallel' or a numeric value if --results-are-fp is set)")
    parser.add_argument('--classifier-results', type=argparse.FileType('rt'), help="Classifier results (if not all the results were provided, the ones that are missing will be obtained with the classifier) whose expected format is: class<tab>src_url<tab>trg_url (class is expected to be 'parallel'/'non-parallel' or a numeric value if --results-are-fp is set)")
    parser.add_argument('--results-are-fp', action='store_true', help="Classification results are FP values intead of 'parallel'/'non-parallel'")
    parser.add_argument('--parallel-threshold', type=float, default=0.5, help="Take URLs as parallel when the score is greater than the provided (only applied when flag --results-are-fp is set)")
    parser.add_argument('--disable-rule-1-1', action='store_true', help="Disable WMT16 rule 1-1")
    parser.add_argument('--disable-near-matchs', action='store_true', help="Disable near-matchs (edition distance)")
    parser.add_argument('--do-not-classify-missing-pairs', action='store_true', help="Missing classified pairs will not be generated")

    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose logging mode")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = initialization()

    utils.set_up_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    logging.debug("Arguments processed: {}".format(str(args)))

    if not args.classifier_command and not args.classifier_results:
        raise Exception("You need to provide either --classifier-command or classifier-results")

    main(args)

