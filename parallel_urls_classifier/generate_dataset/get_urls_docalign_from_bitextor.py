
import os
import sys
import math
import base64
import logging
import argparse

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, f"{cdir}/../..")

import parallel_urls_classifier.utils.utils as utils
import parallel_urls_classifier.generate_dataset.utils_bitextor as utils_bitextor

import numpy as np

_occurrences_warning_only_once = True
_occurrences_warning_already_done = False

def get_doc_nolines_score(src_nolines, trg_nolines, occurrences=-1, src_url=None, trg_url=None):
    global _occurrences_warning_only_once
    global _occurrences_warning_already_done

    if src_nolines < 0 or trg_nolines < 0:
        raise Exception(f"nolines can't be < 0: {src_nolines} - {trg_nolines}")

    max_doc_nolines = max(src_nolines, trg_nolines)
    min_doc_nolines = min(src_nolines, trg_nolines) # always (if provided): occurrences <= min_doc_nolines
    diff = abs(src_nolines - trg_nolines)

    if occurrences >=0 and occurrences > min_doc_nolines:
        if _occurrences_warning_only_once and _occurrences_warning_already_done:
            pass
        else:
            logging.warning("Occurrences > min(src_nolines, trg_nolines): %d > min(%d, %d): this might be possible if --ignore_segmentation was not set in Bifixer "
                            "(re-segmentation might increase the number of occurrences): %s    %s",
                            occurrences, src_nolines, trg_nolines, src_url if src_url else "src_url_not_provided", trg_url if trg_url else "trg_url_not_provided")

            _occurrences_warning_already_done = True

            if _occurrences_warning_only_once:
                logging.warning("Previous warning will only be shown once: you can modify '_occurrences_warning_only_once' for changing this behavior")

    def top_margin(x):
        # WMT16 numbers in order to know how to select a good margin:
        #
        # range_max_nolines: max(src_doc_nolines, trg_doc_nolines)
        #  for a doc in the GS
        # sum_of_diffs: sum(abs(src_doc_nolines - trg_doc_nolines) for
        #  each pair of docs in the range in the GS)
        # avg_diff: sum_of_diffs / range_max_nolines
        #
        # train set:
        #
        #  range_max_nolines  docs_in_range  sum_of_diffs  avg_diff
        #  ----------------------------------------------------------
        #  1 < x <= 2         0              0             0
        #  2 < x <= 4         0              0             0
        #  4 < x <= 8         7              2             0.285714
        #  8 < x <= 16        74             18            0.243243
        #  16 < x <= 32       231            172           0.744589
        #  32 < x <= 64       221            291           1.31674
        #  64 < x <= 128      511            1151          2.25245
        #  128 < x <= 256     398            1954          4.90955
        #  256 < x <= 512     143            1224          8.55944
        #  512 < x <= 1024    28             549           19.6071
        #  1024 < x <= 2048   10             240           24
        #  2048 < x <= 4096   0              0             0
        #  4096 < x <= 8192   1              810           810
        #
        # test set:
        #
        #  range_max_nolines  docs_in_range  sum_of_diffs  avg_diff
        #  ----------------------------------------------------------
        #  1 < x <= 2         1              0             0
        #  2 < x <= 4         0              0             0
        #  4 < x <= 8         3              2             0.666667
        #  8 < x <= 16        33             5             0.151515
        #  16 < x <= 32       193            129           0.668394
        #  32 < x <= 64       601            812           1.35108
        #  64 < x <= 128      650            1684          2.59077
        #  128 < x <= 256     638            1941          3.04232
        #  256 < x <= 512     198            1131          5.71212
        #  512 < x <= 1024    79             13            17.557
        #  1024 < x <= 2048   3              41            13.6667
        #  2048 < x <= 4096   1              87            87
        #  4096 < x <= 8192   2              681           340.5


        # Explanation: every 2 ^ x, x E N, the limit is incresed by 1 -> too much limit
        #return math.log2(x) + 1

        # Explanation: x is a good aproximation, but too much open -> limit multiplying a number < 1 or a function < f(x) = x
        #  1. log2 is a good way of limiting the amount the slope increases
        #  2. log2(x) is too much open -> x * x
        #  3. log2(x * x) + 2 in order to allow that 1 is inside the domain and is inside the increasing domain
        #  4. (log2(x * x) + 2) + 1 in order to allow that 1 is not punished that hard
        # if x is 128 -> limit is 9, which according to intuition and WMT16 pairs of documents is a good value
        #return x / (math.log2(x * x) + 2) + 1

        # Explanation: based in the data from WMT16 -> we use 16 instead of 64 in order to set 75 of score for edge cases
        return x / 16

    if max_doc_nolines == 0:
        m = 1.0 # diff will be 0, so score = 0, but we avoid / 0
    else:
        m = top_margin(max_doc_nolines)

    # Score: the higher, the better
    nolines_score = (1.0 - min(max(diff / m, 0.0), 1.0)) * 100.0 # Domain: [0, 100]; diff / m -> if 1, too much distance -> negative
    occurrences_score = None

    if occurrences >= 0:
        # Fix the score in order to deal with --ignore_segmentation from Bifixer
        # Workaround for fix this issue with awk passing the content through pipe:
        #  awk -F$'\t' 'NR == 1 {print $0} NR > 1 {
        #    a=($8 > 100) ? 100 : $8;
        #    b=($8 <= 100) ? $9 : ($6 + a > 0) ? 2 * (($6 * a) / ($6 + a)): 0;
        #    print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7"\t"a"\t"b}' # if docalign and raw.gz were provided
        occurrences_score = min(occurrences, min_doc_nolines) / min_doc_nolines * 100.0 if min_doc_nolines > 0 else 0.0 # Domain: [0, 100]

    return nolines_score, occurrences_score

def main(args):
    min_occurrences = args.min_occurrences
    bicleaner_threshold = args.bicleaner_threshold
    raw_file = args.raw_file
    raw_file_src_url_idx = args.raw_file_src_url_idx
    raw_file_trg_url_idx = args.raw_file_trg_url_idx
    raw_file_src_text_idx = args.raw_file_src_text_idx
    raw_file_trg_text_idx = args.raw_file_trg_text_idx
    raw_file_bicleaner_idx = args.raw_file_bicleaner_idx
    docalign_mt_matches_files = args.docalign_mt_matches_files
    src_url_files = args.src_url_files
    trg_url_files = args.trg_url_files
    src_sentences_files = args.src_sentences_files
    trg_sentences_files = args.trg_sentences_files
    src_sentences_preprocess_cmd = args.src_sentences_preprocess_cmd
    trg_sentences_preprocess_cmd = args.trg_sentences_preprocess_cmd
    raw_preprocess_cmd = args.raw_preprocess_cmd
    docalign_threshold = args.docalign_threshold
    process_docalign = True if docalign_mt_matches_files else False
    n_jobs = args.n_jobs

    if not docalign_mt_matches_files:
        docalign_mt_matches_files = []

    if len(src_url_files) != len(src_sentences_files):
        raise Exception("Unexpected different number of source files provided for url.gz and sentences.gz " \
                        f"({len(src_url_files)} vs {len(src_sentences_files)})")
    if len(trg_url_files) != len(trg_sentences_files):
        raise Exception("Unexpected different number of target files provided for url.gz and sentences.gz " \
                        f"({len(trg_url_files)} vs {len(trg_sentences_files)})")
    if not process_docalign and not sent_file:
        raise Exception("You need to provide either docalign files, sent.gz file or both of them")

    urls_shard_batch = {"src": {}, "trg": {}}
    sentences_shard_batch = {"src": {}, "trg": {}}
    docalign_mt_matches_shard_batches = []
    docalign_mt_matches_shard_quantity = {}
    shards = set()

    # Get shard and batches for url.gz and sentences.gz
    for url_files, sentences_files, direction in [(src_url_files, src_sentences_files, "src"),
                                                  (trg_url_files, trg_sentences_files, "trg")]:
        for url_file, sentences_file in zip(url_files, sentences_files):
            url_data = url_file.split('/')
            sentences_data = sentences_file.split('/')
            url_shard = url_data[-3]
            url_batch = url_data[-2]
            sentences_shard = sentences_data[-3]
            sentences_batch = sentences_data[-2]

            if url_shard not in urls_shard_batch[direction]:
                urls_shard_batch[direction][url_shard] = {}
            if sentences_shard not in sentences_shard_batch[direction]:
                sentences_shard_batch[direction][sentences_shard] = {}

            shards.add(url_shard)
            shards.add(sentences_shard)

            urls_shard_batch[direction][url_shard][url_batch] = url_file
            sentences_shard_batch[direction][sentences_shard][sentences_batch] = sentences_file

    if process_docalign:
        # Get shard and batches for matches
        for docalign_mt_matches_file in docalign_mt_matches_files:
            data = docalign_mt_matches_file.split('/')
            shard = data[-2]
            data = data[-1].split('.')[0].split('_')
            src_batch = data[0][2:]
            trg_batch = data[1][2:]

            if shard not in shards:
                raise Exception(f"Shard found in matches but not in url.gz/sentences.gz files: {shard}")

            docalign_mt_matches_shard_batches.append((shard, (src_batch, trg_batch)))

            if shard not in docalign_mt_matches_shard_quantity:
                docalign_mt_matches_shard_quantity[shard] = 0

            docalign_mt_matches_shard_quantity[shard] += 1

        # Docalign sanity check
        for shard in shards:
            urls_quantity = len(urls_shard_batch["src"][shard]) * len(urls_shard_batch["trg"][shard])

            if urls_quantity != docalign_mt_matches_shard_quantity[shard]:
                raise Exception("Unexpected quantity of url.gz/sentences.gz files taking into account matches files " \
                                f"(shard {shard}: {urls_quantity} vs {docalign_mt_matches_shard_quantity[shard]})")

    docalign_src_urls, docalign_trg_urls = {}, {}
    docalign_url_scores = {}
    total_printed_urls = 0
    total_possible_printed_urls = 0

    # Get number of lines per document/URL
    src_urls_statistics = \
        utils_bitextor.get_statistics_from_url_and_sentences(src_url_files, src_sentences_files, preprocess_cmd=src_sentences_preprocess_cmd,
                                                             n_jobs=n_jobs)
    trg_urls_statistics = \
        utils_bitextor.get_statistics_from_url_and_sentences(trg_url_files, trg_sentences_files, preprocess_cmd=trg_sentences_preprocess_cmd,
                                                             n_jobs=n_jobs)

    logging.info("Number of URLs (src, trg): (%d, %d)", len(src_urls_statistics), len(trg_urls_statistics))

    # Print header
    sys.stdout.write("src_url\ttrg_url")

    if process_docalign:
        sys.stdout.write("\tdocalign_score")

    sys.stdout.write("\tsrc_doc_nolines\ttrg_doc_nolines\tsrc_and_trg_docs_nolines_score\tsrc_doc_tokens\ttrg_doc_tokens")

    if raw_file:
        sys.stdout.write("\tsegalign_src_and_trg_nolines\tsegalign_src_and_trg_nolines_score\tsegalign_and_docs_nolines_score_f1")

        if raw_file_bicleaner_idx is not None:
            sys.stdout.write("\tavg_doc_bicleaner_score")

        sys.stdout.write("\tsrc_doc_alignment_tokens\ttrg_doc_alignment_tokens\ttokens_score")

    sys.stdout.write('\n')

    if process_docalign:
        # Get aligned URLs from matches
        for docalign_mt_matches_file, docalign_mt_matches_shard_batch in zip(docalign_mt_matches_files, docalign_mt_matches_shard_batches):
            src_urls_idx, trg_urls_idx, scores = [], [], []

            with utils.open_xz_or_gzip_or_plain(docalign_mt_matches_file) as f:
                logging.debug("Processing '%s'", docalign_mt_matches_file)

                for line in f:
                    line = line.strip().split('\t')
                    score = float(line[0])
                    src_url_idx = int(line[1])
                    trg_url_idx = int(line[2])

                    if score < docalign_threshold:
                        continue

                    scores.append(score)
                    src_urls_idx.append(src_url_idx)
                    trg_urls_idx.append(trg_url_idx)

            src_urls = utils_bitextor.get_urls_from_idxs(
                        urls_shard_batch["src"][docalign_mt_matches_shard_batch[0]][docalign_mt_matches_shard_batch[1][0]],
                        src_urls_idx)
            trg_urls = utils_bitextor.get_urls_from_idxs(
                        urls_shard_batch["trg"][docalign_mt_matches_shard_batch[0]][docalign_mt_matches_shard_batch[1][1]],
                        trg_urls_idx)

            if len(src_urls) != len(trg_urls):
                raise Exception("Unexpected different number of src and trg URLs "
                                f"({len(src_urls)} vs {len(trg_urls)}) in file '{docalign_mt_matches_file}'")
            if len(src_urls) != len(scores):
                raise Exception("Unexpected different number of URLs and scores "
                                f"({len(src_urls)} vs {len(scores)}) in file '{docalign_mt_matches_file}'")

            logging.debug("Number of aligned URLs from '%s': %d", docalign_mt_matches_file, len(src_urls))

            for score, src_url, trg_url in zip(scores, src_urls, trg_urls):
                try:
                    docalign_src_urls[src_url] += 1
                except KeyError:
                    docalign_src_urls[src_url] = 1
                try:
                    docalign_trg_urls[trg_url] += 1
                except KeyError:
                    docalign_trg_urls[trg_url] = 1

                k = hash(f"{src_url}\t{trg_url}")
                docalign_url_scores[k] = score
                src_url_nolines = src_urls_statistics[src_url]["nolines"]
                trg_url_nolines = trg_urls_statistics[trg_url]["nolines"]
                src_url_tokens = src_urls_statistics[src_url]["tokens"]
                trg_url_tokens = trg_urls_statistics[trg_url]["tokens"]
                nolines_score, _ = get_doc_nolines_score(src_url_nolines, trg_url_nolines)

                if not raw_file:
                    # We want to avoid scientific notation
                    score = round(score, 4)
                    nolines_score = round(nolines_score, 4)

                    sys.stdout.write(f"{src_url}\t{trg_url}\t{score}\t{src_url_nolines}\t{trg_url_nolines}\t{nolines_score}"
                                     f"\t{src_url_tokens}\t{trg_url_tokens}")
                    sys.stdout.write('\n')

                    total_printed_urls += 1

    # Process raw.gz file
    if raw_file:
        logging.info("Processing raw.gz file")

        aligned_urls = utils_bitextor.get_statistics_from_raw(raw_file, raw_file_src_url_idx, raw_file_trg_url_idx,
                                                              raw_file_src_text_idx, raw_file_trg_text_idx,
                                                              bicleaner_idx=raw_file_bicleaner_idx, preprocess_cmd=raw_preprocess_cmd)

        logging.info("Unique different paired URLs: %d", len(aligned_urls))

        skipped_bc_docalign = 0
        skipped_bc_min_occ = 0
        skipped_bc_bicleaner = 0
        aligned_tokens = {}

        for idx, (url, data) in enumerate(aligned_urls.items()):
            # Iterate through unique URLs
            occurrences = data["occurrences"]
            avg_doc_bicleaner_score = min(data["bicleaner_sum"] / occurrences, 1.0)

            if (idx + 1) % 10000 == 0:
                logging.debug("%.2f finished", (idx + 1) * 100.0 / len(aligned_urls))
                logging.debug("Currently skipped URLs (min occ., docalign, bicleaner): (%d, %d, %d)",
                              skipped_bc_min_occ, skipped_bc_docalign, skipped_bc_bicleaner)

            if occurrences < min_occurrences:
                skipped_bc_min_occ += 1
                continue
            if raw_file_bicleaner_idx is not None and avg_doc_bicleaner_score < bicleaner_threshold:
                skipped_bc_bicleaner += 1
                continue

            urls = url.split('\t')

            assert len(urls) == 2, "Error"

            src_url = urls[0]
            trg_url = urls[1]
            k = hash(f"{src_url}\t{trg_url}")

            if process_docalign:
                try:
                    docalign_src_urls[src_url]
                    docalign_trg_urls[trg_url]
                except KeyError:
                    skipped_bc_docalign += 1
                    continue

            try:
                src_url_nolines = src_urls_statistics[src_url]["nolines"]
                trg_url_nolines = trg_urls_statistics[trg_url]["nolines"]
                src_url_tokens = src_urls_statistics[src_url]["tokens"]
                trg_url_tokens = trg_urls_statistics[trg_url]["tokens"]
            except KeyError:
                logging.warning("src URL (%s) or trg URL (%s) not in aligned URLs", src_url, trg_url)

                continue

            score = 0.0
            nolines_score, occurrences_score = get_doc_nolines_score(src_url_nolines, trg_url_nolines, occurrences=occurrences,
                                                                     src_url=src_url, trg_url=trg_url)
            nolines_and_occurences_score_f1 = 2 * ((nolines_score * occurrences_score) / (nolines_score + occurrences_score)) if not np.isclose(nolines_score + occurrences_score, 0.0) else 0.0
            aligned_src_tokens = aligned_urls[src_url]["src_tokens"]
            aligned_trg_tokens = aligned_urls[trg_url]["trg_tokens"]
            tokens_score = (aligned_src_tokens + aligned_trg_tokens) / (src_url_tokens + trg_url_tokens)

            try:
                score = docalign_url_scores[k]
            except KeyError:
                score = -1.0

                if process_docalign:
                    logging.warning("Docalign score not found for URLs: ('%s', '%s')", src_url, trg_url)

            # We want to avoid scientific notation
            score = round(score, 4)
            nolines_score = round(nolines_score, 4)
            occurrences_score = round(occurrences_score, 4)
            nolines_and_occurences_score_f1 = round(nolines_and_occurences_score_f1, 4)
            avg_doc_bicleaner_score = round(avg_doc_bicleaner_score, 4)
            tokens_score = round(tokens_score, 4)

            sys.stdout.write(f"{src_url}\t{trg_url}")

            if process_docalign:
                sys.stdout.write(f"\t{score}")

            sys.stdout.write(f"\t{src_url_nolines}\t{trg_url_nolines}\t{nolines_score}\t{src_url_tokens}\t{trg_url_tokens}")
            sys.stdout.write(f"\t{occurrences}\t{occurrences_score}\t{nolines_and_occurences_score_f1}")

            if raw_file_bicleaner_idx is not None:
                sys.stdout.write(f"\t{avg_doc_bicleaner_score}")

            sys.stdout.write(f"\t{aligned_src_tokens}\t{aligned_trg_tokens}\t{tokens_score}")
            sys.stdout.write('\n')

            total_printed_urls += 1

        logging.info("Total skipped URLs (min occ., docalign, bicleaner): (%d, %d, %d)",
                     skipped_bc_min_occ, skipped_bc_docalign, skipped_bc_bicleaner)

    logging.info("Total printed URLs: %d", total_printed_urls)

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Get aligned URLs")

    parser.add_argument('--docalign-mt-matches-files', nargs='+', help=".matches files")
    parser.add_argument('--src-url-files', nargs='+', required=True, help="Source url.gz files from sharding")
    parser.add_argument('--trg-url-files', nargs='+', required=True, help="Target url.gz files from sharding")
    parser.add_argument('--src-sentences-files', nargs='+', required=True, help="Source sentences.gz files from sharding")
    parser.add_argument('--trg-sentences-files', nargs='+', required=True, help="Target sentences.gz files from sharding")
    parser.add_argument('--src-sentences-preprocess-cmd',
                        help="Preprocess command to apply to the src sentences."
                             "The provided command has to read sentences from stdin and print to stdout")
    parser.add_argument('--trg-sentences-preprocess-cmd',
                        help="Preprocess command to apply to the trg sentences."
                             "The provided command has to read sentences from stdin and print to stdout")
    parser.add_argument('--raw-file', help=".rwa.gz file. If not provided, only docalign will be taken into account")
    parser.add_argument('--raw-file-src-url-idx', type=int, default=0, help=".raw.gz file src URL idx")
    parser.add_argument('--raw-file-trg-url-idx', type=int, default=1, help=".raw.gz file trg URL idx")
    parser.add_argument('--raw-file-src-text-idx', type=int, default=2, help=".raw.gz file text URL idx")
    parser.add_argument('--raw-file-trg-text-idx', type=int, default=3, help=".raw.gz file text URL idx")
    parser.add_argument('--raw-file-bicleaner-idx', type=int, default=None, help=".raw.gz file bicleaner idx")
    parser.add_argument('--raw-preprocess-cmd',
                        help="Preprocess command to apply to the src and trg alignments."
                             "The provided command has to read pair of sentences separated by tab from stdin and print to stdout")
    parser.add_argument('--n-jobs', type=int, default=-1, help="Number of parallel jobs to use (-n means to use all CPUs - n + 1)")

    parser.add_argument('--min-occurrences', type=int, default=0, help="Min. occurrences of URLs pairs")
    parser.add_argument('--bicleaner-threshold', type=float, default=0.0,
                        help="Bicleaner threshold. The threshold is applied to the avg scores for all the sentences of the document")
    parser.add_argument('--docalign-threshold', type=float, default=0.0, help="Docalign threshold")

    parser.add_argument('-q', '--quiet', action='store_true', help="Silent logging mode")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = initialization()

    utils.set_up_logging(level=logging.INFO if args.quiet else logging.DEBUG)

    logging.debug("Arguments processed: {}".format(str(args)))

    main(args)
