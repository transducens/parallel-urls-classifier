
import base64
import logging
import argparse
import itertools
import subprocess

from ..utils import levenshtein
from ..utils import utils

def process_pairs(pairs, command):
    results = []

    sp_result = subprocess.Popen(f"{command} | cut -f1", shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    aux_result, _ = sp_result.communicate('\n'.join(pairs).encode())

    results = list(map(lambda r: r == "parallel", aux_result.decode("utf-8").strip().split('\n')))

    return results

def evaluate_recall(src_pairs, trg_pairs, src_gs_pairs, trg_gs_pairs, src_urls, trg_urls, src_docs, trg_docs):
    tp = 0
    seen_src_pairs, seen_trg_pairs = set(), set()
    gs_pairs = set(f"{src_gs_pair}\t{trg_gs_pair}" for src_gs_pair, trg_gs_pair in zip(src_gs_pairs, trg_gs_pairs))

    for src_pair, trg_pair in itertools.product(src_pairs, trg_pairs):
        pair = f"{src_pair}\t{trg_pair}"

        if pair in gs_pairs and src_pair not in seen_src_pairs and trg_pair not in seen_trg_pairs:
            tp += 1
        else:
            if src_pair in src_gs_pairs and trg_pair in src_gs_pairs:
                pass
            else:
                # Near-matches
                near_match_src = src_pair in src_gs_pairs and trg_pair not in trg_gs_pairs and src_pair not in seen_src_pairs and trg_pair not in seen_trg_pairs
                near_match_trg = trg_pair in trg_gs_pairs and src_pair not in src_gs_pairs and trg_pair not in seen_trg_pairs and src_pair not in seen_src_pairs

                if near_match_src and near_match_trg:
                    pass
                elif near_match_src or near_match_trg:
                    if near_match_src:
                        doc_1_idx = trg_urls.index(trg_pair)
                        doc_2_idx = trg_urls.index(trg_gs_pairs[src_gs_pairs.index(src_pair)])
                        doc_1 = trg_docs[doc_1_idx]
                        doc_2 = trg_docs[doc_2_idx]
                        url_1 = trg_pair
                        url_2 = trg_gs_pairs[src_gs_pairs.index(src_pair)]
                    else:
                        doc_1_idx = src_urls.index(src_pair)
                        doc_2_idx = src_urls.index(src_gs_pairs[trg_gs_pairs.index(trg_pair)])
                        doc_1 = src_docs[doc_1_idx]
                        doc_2 = src_docs[doc_2_idx]
                        url_1 = src_pair
                        url_2 = src_gs_pairs[trg_gs_pairs.index(trg_pair)]

                    similarity = levenshtein.levenshtein_opt_space_and_band(doc_1, doc_2, nfactor=max(len(doc_1), len(doc_2)), percentage=0.06)["similarity"]

                    logging.debug("Near-match similarity (url_1, url_2, similarity_score): ('%s', '%s', %f)", url_1, url_2, similarity)

                    if similarity >= 0.95:
                        logging.debug("Near-match found")

                        tp += 1

        seen_src_pairs.add(src_pair)
        seen_trg_pairs.add(trg_pair)

    logging.info("True positives: %d", tp)
    logging.info("GS pairs: %d", len(gs_pairs))

    recall = tp / len(gs_pairs)

    print(f"Recall: {recall}")

def main(args):
    input_file = args.input_file
    gold_standard_file = args.gold_standard_file
    classifier_command = args.classifier_command
    batch_size = args.batch_size

    src_urls, trg_urls = [], []
    src_docs, trg_docs = [], []
    parallel = []
    domain = None

    for line in input_file:
        lang, url, doc = line.rstrip('\n').split('\t')
        d = url.lstrip("http://").lstrip("https://").split('/')[0]

        if domain is None:
            domain = d
        if d != domain:
            raise Exception(f"Provided different domain: '{d}' vs '{domain}'")

        if lang not in ("en", "fr"):
            raise Exception(f"Unexpected lang (expected: en, fr): {lang}")

        doc = base64.b64decode(doc).decode("utf-8").strip()

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
        src_domain = src.lstrip("http://").lstrip("https://").split('/')[0]
        trg_domain = trg.lstrip("http://").lstrip("https://").split('/')[0]

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
    for idx, (src_url, trg_url) in enumerate(itertools.product(src_urls, trg_urls), 1):
        pairs.append(f"{src_url}\t{trg_url}")

        if len(pairs) >= batch_size:
            logging.info("Batch size %d", idx // batch_size)

            parallel.extend(process_pairs(pairs, classifier_command))

            pairs = []

    if len(pairs) != 0:
        logging.info("Batch size %f", idx / batch_size)

        parallel.extend(process_pairs(pairs, classifier_command))

        pairs = []

    expected_values = len(src_urls) * len(trg_urls)

    assert expected_values  == len(parallel), f"Unexpected parallel length: {expected_values} vs {len(parallel)}"

    parallel_values = parallel.count(True)
    non_parallel_values = parallel.count(False)

    logging.info(f"(parallel, non-parallel): (%d, %d)", parallel_values, non_parallel_values)

    for v, (src_url, trg_url) in zip(parallel, itertools.product(src_urls, trg_urls)):
        logging.debug(f"{'parallel' if v else 'non-parallel'}\t{src_url}\t{trg_url}")

    src_pairs = [url[0] for p, url in zip(parallel, itertools.product(src_urls, trg_urls)) if p]
    trg_pairs = [url[1] for p, url in zip(parallel, itertools.product(src_urls, trg_urls)) if p]

    assert len(src_pairs) == len(trg_pairs), f"Different src and trg parallel URLs: {len(src_pairs)} vs {len(trg_pairs)}"
    assert len(src_pairs) == parallel_values, f"Unexpected quantity of parallel values: {len(src_pairs)} vs {parallel_values}"

    evaluate_recall(src_pairs, trg_pairs, src_gs, trg_gs, src_urls, trg_urls, src_docs, trg_docs)

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="WMT16 evaluation")

    parser.add_argument('input_file', type=argparse.FileType('rt'), help="Input file with the following format: lang<tab>URL<tab>base64-doc")
    parser.add_argument('gold_standard_file', type=argparse.FileType('rt'), help="Gold standard file")

    parser.add_argument('--classifier-command', required=True, help="Classifier command")
    parser.add_argument('--batch-size', type=int, default=256, help="Batch size")
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose logging mode")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = initialization()

    utils.set_up_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    logging.debug("Arguments processed: {}".format(str(args)))

    main(args)

