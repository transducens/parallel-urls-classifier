
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

_occurrences_warning_only_once = False
_occurrences_warning_already_done = False
logger = logging.getLogger("parallel_urls_classifier")

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
            logger.warning("Occurrences > min(src_nolines, trg_nolines): %d > min(%d, %d): this might be possible if --ignore_segmentation was not set in Bifixer "
                           "(re-segmentation might increase the number of occurrences) or different versions of the documents were provided: %s\t%s",
                           occurrences, src_nolines, trg_nolines, src_url if src_url else "src_url_not_provided", trg_url if trg_url else "trg_url_not_provided")

            _occurrences_warning_already_done = True

            if _occurrences_warning_only_once:
                logger.warning("Previous warning will only be shown once: you can modify '_occurrences_warning_only_once' in order to change this behavior")

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
    # This score might not be as good as should be due to, for instance, boilerplate removal (it might be different for each direction)
    nolines_score = (1.0 - min(max(diff / m, 0.0), 1.0)) # Domain: [0, 1]; diff / m -> if 1, too much distance -> negative
    occurrences_score = None

    if occurrences >= 0:
        # Fix the score in order to deal with --ignore_segmentation from Bifixer
        # Workaround for fix this issue with awk passing the content through pipe:
        #  awk -F$'\t' 'NR == 1 {print $0} NR > 1 {
        #    a=($8 > 100) ? 100 : $8;
        #    b=($8 <= 100) ? $9 : ($6 + a > 0) ? 2 * (($6 * a) / ($6 + a)): 0;
        #    print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7"\t"a"\t"b}' # if docalign and segalign files were provided
        # This score might not be as good as should be since Bleualign runs the gap filler feature, and this will lower the score
        occurrences_score = min(occurrences, min_doc_nolines) / min_doc_nolines if min_doc_nolines > 0 else 0.0 # Domain: [0, 1]

    return nolines_score, occurrences_score

_tokens_score_warning_only_once = False
_src_tokens_warning_already_done = False
_trg_tokens_warning_already_done = False
def get_aligned_tokens_score(src_url_tokens, trg_url_tokens, aligned_src_tokens, aligned_trg_tokens,
                             src_url=None, trg_url=None, check=True):
    global _tokens_score_warning_only_once
    global _src_tokens_warning_already_done
    global _trg_tokens_warning_already_done

    aligned_tokens = min(src_url_tokens, aligned_src_tokens) + min(trg_url_tokens, aligned_trg_tokens)

    # Checks
    # Known problem: since Bleualign applies gap filler, this might lead to situations where the tokenization of the segalign sentences have
    #  more tokens than expected, for instance:
    #  - Src doc sentences: this is a sentence. . .\nthis is another sentence
    #  - Trg doc sentences: a\nb
    #  - Segalign: this is a sentence. . .\ta\nthis is another sentence\tb
    #  - Src doc tokenized: this is a sentence. . .\nthis is another sentence
    #  - Trg doc tokenized: a\nb
    #  - Segalign tokenized: this is a sentence . . .\ta\nthis is another sentence\tb # Here is the problem!
    #  Problem: The '.' has been tokenized in the segalign but not in the src doc because in the src doc has been detected as ellipsis, but
    #   in the segalign has not been detected after apply gap filler

    if check and aligned_src_tokens > src_url_tokens:
        if _tokens_score_warning_only_once and _src_tokens_warning_already_done:
            pass
        else:
            logger.warning("aligned_src_tokens > src_url_tokens: %d > %d: this might be possible if some preprocessing was applied with Bifixer "
                           "or different versions of the documents were provided: %s\t%s",
                           aligned_src_tokens, src_url_tokens, src_url if src_url else "src_url_not_provided", trg_url if trg_url else "trg_url_not_provided")

            _src_tokens_warning_already_done = True

            if _tokens_score_warning_only_once:
                logger.warning("Previous warning will only be shown once: you can modify '_tokens_score_warning_only_once' in order to change this behavior")

    if check and aligned_trg_tokens > trg_url_tokens:
        if _tokens_score_warning_only_once and _trg_tokens_warning_already_done:
            pass
        else:
            logger.warning("aligned_trg_tokens > trg_url_tokens: %d > %d: this might be possible if some preprocessing was applied with Bifixer "
                           "or different versions of the documents were provided: %s\t%s",
                           aligned_trg_tokens, trg_url_tokens, src_url if src_url else "src_url_not_provided", trg_url if trg_url else "trg_url_not_provided")

            _trg_tokens_warning_already_done = True

            if _tokens_score_warning_only_once:
                logger.warning("Previous warning will only be shown once: you can modify '_tokens_score_warning_only_once' in order to change this behavior")

    # Get score
    tokens_score = aligned_tokens / (src_url_tokens + trg_url_tokens)

    return tokens_score

_log_read_pairs = 10000 # segalign files
def main(args):
    min_occurrences = args.min_occurrences
    segalign_files = args.segalign_files
    segalign_files_src_url_idx = args.segalign_files_src_url_idx
    segalign_files_trg_url_idx = args.segalign_files_trg_url_idx
    segalign_files_src_text_idx = args.segalign_files_src_text_idx
    segalign_files_trg_text_idx = args.segalign_files_trg_text_idx
    segalign_files_segalign_score_idx = args.segalign_files_segalign_score_idx
    segalign_files_bicleaner_score_idx = args.segalign_files_bicleaner_score_idx
    docalign_mt_matches_files = args.docalign_mt_matches_files
    src_url_files = args.src_url_files
    trg_url_files = args.trg_url_files
    src_sentences_files = args.src_sentences_files
    trg_sentences_files = args.trg_sentences_files
    src_sentences_preprocess_cmd = args.src_sentences_preprocess_cmd
    trg_sentences_preprocess_cmd = args.trg_sentences_preprocess_cmd
    segalign_preprocess_cmd = args.segalign_preprocess_cmd
    docalign_threshold = args.docalign_threshold
    n_jobs = args.n_jobs # Experiments without preprocess_cmd: {  0: 5:45,  1: 6.17,  2: 3.00,  3: 2.08,  4: 1.37,  5: 1.20,
                         #                                        6: 1.07,  7: 1.01,  8: 0.54,  9: 0.47, 10: 0.51, 11: 0.48,
                         #                                       12: 0.45, 13: 0.36, 14: 0.35, 15: 0.34, 16: 0.33, 17: 0.31,
                         #                                       18: 0.30, 19: 0.31, 20: 0.31, 21: 0.30, 22: 0.30, 23: 0.29,
                         #                                       24: 0.28, 25: 0.28, 26: 0.29, 27: 0.30, 28: 0.30, 29: 0.32,
                         #                                       30: 0.33, 31: 0.32, 32: 0.35 }
    ignore_duplicated_urls = args.ignore_duplicated_urls
    parallelize = n_jobs != 0

    if not segalign_files:
        segalign_files = []

    if len(src_url_files) != len(src_sentences_files):
        raise Exception("Unexpected different number of source files provided for url.gz and sentences.gz " \
                        f"({len(src_url_files)} vs {len(src_sentences_files)})")
    if len(trg_url_files) != len(trg_sentences_files):
        raise Exception("Unexpected different number of target files provided for url.gz and sentences.gz " \
                        f"({len(trg_url_files)} vs {len(trg_sentences_files)})")

    urls_shard_batch = {"src": {}, "trg": {}}
    sentences_shard_batch = {"src": {}, "trg": {}}
    shards = set()
    docalign_mt_matches_shard_batches = {}
    segalign_shard_batches = {}
    segalign_shard_batches_rev = {}

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

    def get_transient_shards_and_batches_info_from_path(desc, files, shard_batches, sanity_check=True):
        # Get shard and batches for provided files (e.g. matches, segalign)

        shard_quantity = {}

        for file in files:
            data = file.split('/')
            shard = data[-2]
            data = data[-1].split('.')[0].split('_')
            src_batch = data[0][2:]
            trg_batch = data[1][2:]

            if shard not in shards:
                raise Exception(f"Shard found in {desc} but not in url.gz/sentences.gz files: {shard}")

            shard_data = (shard, (src_batch, trg_batch))

            if file in shard_batches:
                logger.warning("Same %s file provided multiple times: %s", desc, file)

                if shard_batches[file] != shard_data:
                    raise Exception(f"The {desc} file contains different sharding data (bug?): {shard_data} vs {shard_batches[file]}")
            else:
                shard_batches[file] = shard_data

            if shard not in shard_quantity:
                shard_quantity[shard] = 0

            shard_quantity[shard] += 1

        if files and sanity_check:
            for shard in shards:
                urls_quantity = len(urls_shard_batch["src"][shard]) * len(urls_shard_batch["trg"][shard])

                if urls_quantity != shard_quantity[shard]:
                    raise Exception(f"Unexpected quantity of url.gz/sentences.gz files taking into account {desc} files: " \
                                    f"shard {shard}: {urls_quantity} vs {shard_quantity[shard]}")

    # Update and check shards and batches from docalign and segalign files
    get_transient_shards_and_batches_info_from_path("matches", docalign_mt_matches_files, docalign_mt_matches_shard_batches)
    get_transient_shards_and_batches_info_from_path("segalign", segalign_files, segalign_shard_batches)

    # Get reverse index for the segalign files and the sharding data
    for file, shard_batches in segalign_shard_batches.items():
        shard, (src_batch, trg_batch) = shard_batches

        if shard not in segalign_shard_batches_rev:
            segalign_shard_batches_rev[shard] = {}
        if src_batch not in segalign_shard_batches_rev[shard]:
            segalign_shard_batches_rev[shard][src_batch] = {}
        if trg_batch in segalign_shard_batches_rev[shard][src_batch]:
            raise Exception("Same shard, src and trg batch provided for multiple segalign files")

        segalign_shard_batches_rev[shard][src_batch][trg_batch] = file

    docalign_src_urls, docalign_trg_urls = {}, {}
    docalign_url_scores = {}
    total_printed_urls = 0
    total_possible_printed_urls = 0
    processed_docalign_files = set()
    processed_segalign_files = set() # subset of processed_docalign_files (sharding data)

    # Get number of lines per document/URL
    logger.debug("Processing src url.gz and sentences.gz files")
    src_urls_statistics, src_urls_skipped = \
        utils_bitextor.get_statistics_from_url_and_sentences(src_url_files, src_sentences_files, preprocess_cmd=src_sentences_preprocess_cmd,
                                                             parallelize=parallelize, n_jobs=n_jobs)
    logger.debug("Processing trg url.gz and sentences.gz files")
    trg_urls_statistics, trg_urls_skipped = \
        utils_bitextor.get_statistics_from_url_and_sentences(trg_url_files, trg_sentences_files, preprocess_cmd=trg_sentences_preprocess_cmd,
                                                             parallelize=parallelize, n_jobs=n_jobs)

    logger.info("Number of URLs (src, trg): (%d, %d)", len(src_urls_statistics), len(trg_urls_statistics))
    logger.info("Number of URLs that have been detected as duplicated (src, trg): (%d, %d)", len(src_urls_skipped), len(trg_urls_skipped))

    # Print header
    sys.stdout.write("src_url\ttrg_url\tdocalign_score")
    sys.stdout.write("\tsrc_doc_nolines\ttrg_doc_nolines\tsrc_and_trg_docs_nolines_score\tsrc_doc_tokens\ttrg_doc_tokens")

    if segalign_files:
        sys.stdout.write("\tsegalign_src_and_trg_nolines\tsegalign_src_and_trg_nolines_score\tsegalign_and_docs_nolines_score_f1")
        sys.stdout.write("\tsrc_doc_alignment_tokens\ttrg_doc_alignment_tokens\ttokens_score")
        sys.stdout.write("\tsrc_doc_alignment_tokens_weighted_segalign\ttrg_doc_alignment_tokens_weighted_segalign\ttokens_score_weighted_segalign")

        if segalign_files_bicleaner_score_idx is not None:
            sys.stdout.write("\tsrc_doc_alignment_tokens_weighted_bicleaner\ttrg_doc_alignment_tokens_weighted_bicleaner\ttokens_score_weighted_bicleaner")
            sys.stdout.write("\tsrc_doc_alignment_tokens_weighted_segalign_and_bicleaner\ttrg_doc_alignment_tokens_weighted_segalign_and_bicleaner\ttokens_score_weighted_segalign_and_bicleaner")

    sys.stdout.write('\n')

    # Get aligned URLs from matches and, optionally, process segalign
    for idx, docalign_mt_matches_file in enumerate(docalign_mt_matches_files, 1):
        src_urls_idx, trg_urls_idx, scores = [], [], []
        shard, (src_batch, trg_batch) = docalign_mt_matches_shard_batches[docalign_mt_matches_file]

        with utils.open_xz_or_gzip_or_plain(docalign_mt_matches_file) as fd:
            logger.debug("Processing docalign file #%s (out of %d): %s", idx, len(docalign_mt_matches_files), docalign_mt_matches_file)

            for line in fd:
                line = line.strip().split('\t')
                score = float(line[0])
                src_url_idx = int(line[1])
                trg_url_idx = int(line[2])

                if score < docalign_threshold:
                    continue

                scores.append(score)
                src_urls_idx.append(src_url_idx)
                trg_urls_idx.append(trg_url_idx)

        src_urls = utils_bitextor.get_urls_from_idxs(urls_shard_batch["src"][shard][src_batch], src_urls_idx)
        trg_urls = utils_bitextor.get_urls_from_idxs(urls_shard_batch["trg"][shard][trg_batch], trg_urls_idx)

        if len(src_urls) != len(trg_urls):
            raise Exception("Unexpected different number of src and trg URLs "
                            f"({len(src_urls)} vs {len(trg_urls)}) in file '{docalign_mt_matches_file}'")
        if len(src_urls) != len(scores):
            raise Exception("Unexpected different number of URLs and scores "
                            f"({len(src_urls)} vs {len(scores)}) in file '{docalign_mt_matches_file}'")

        logger.debug("Aligned URLs from docalign file: %s: %d", docalign_mt_matches_file, len(src_urls))

        # Process docalign file
        for idx_docalign, (score, src_url, trg_url) in enumerate(zip(scores, src_urls, trg_urls), 1):
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

            if not segalign_files:
                if ignore_duplicated_urls:
                    _skip = False

                    if src_url in src_urls_skipped:
                        logger.debug("Src URL #%d ignored: %s", idx_docalign, src_url)

                        _skip = True
                    if trg_url in trg_urls_skipped:
                        logger.debug("Trg URL #%d ignored: %s", idx_docalign, trg_url)

                        _skip = True

                    if _skip:
                        continue

                src_url_nolines = src_urls_statistics[src_url]["nolines"]
                trg_url_nolines = trg_urls_statistics[trg_url]["nolines"]
                src_url_tokens = src_urls_statistics[src_url]["tokens"]
                trg_url_tokens = trg_urls_statistics[trg_url]["tokens"]
                nolines_score, _ = get_doc_nolines_score(src_url_nolines, trg_url_nolines)

                # We want to avoid scientific notation
                score = round(score, 4)
                nolines_score = round(nolines_score, 4)

                sys.stdout.write(f"{src_url}\t{trg_url}\t{score}\t{src_url_nolines}\t{trg_url_nolines}\t{nolines_score}"
                                    f"\t{src_url_tokens}\t{trg_url_tokens}")
                sys.stdout.write('\n')

                total_printed_urls += 1

        processed_docalign_files.add(docalign_mt_matches_file)

        # Process segalign file
        if segalign_files:
            try:
                segalign_file = segalign_shard_batches_rev[shard][src_batch][trg_batch]
            except KeyError:
                logger.error("Segalign file from the previous docalign file processed seems to don't be available: shard, src and trg batch: "
                             "%d %d %d", shard, src_batch, trg_batch)

                continue

            logger.info("Processing segalign file #%d: %s", idx, segalign_file)

            # Segalign files shouldn't have been post processed and the data should be the same that content from sentences.gz
            aligned_urls = utils_bitextor.get_statistics_from_segalign(segalign_file, segalign_files_src_url_idx, segalign_files_trg_url_idx,
                                                                       segalign_files_src_text_idx, segalign_files_trg_text_idx,
                                                                       segalign_files_segalign_score_idx, preprocess_cmd=segalign_preprocess_cmd,
                                                                       parallelize=parallelize, n_jobs=n_jobs,
                                                                       segalign_files_bicleaner_score_idx=segalign_files_bicleaner_score_idx)

            logger.info("Unique different URL pairs: %d", len(aligned_urls))
            logger.debug("Documents with 0 sentences aligned: %d", len(src_urls) - len(aligned_urls))

            skipped_bc_docalign = 0
            skipped_bc_min_occ = 0
            skipped_bc_duplicated = 0
            aligned_tokens = {}

            for idx_aligned_urls, (url, data) in enumerate(aligned_urls.items()):
                # Iterate through unique URLs
                occurrences = data["occurrences"]

                if (idx_aligned_urls + 1) % _log_read_pairs == 0:
                    logger.debug("%.2f finished", (idx_aligned_urls + 1) * 100.0 / len(aligned_urls))
                    logger.debug("Currently skipped URLs (min occ., docalign, duplicated): (%d, %d, %d)",
                                skipped_bc_min_occ, skipped_bc_docalign, skipped_bc_duplicated)

                if occurrences < min_occurrences:
                    skipped_bc_min_occ += 1
                    continue

                urls = url.split('\t')

                assert len(urls) == 2, "Error"

                src_url = urls[0]
                trg_url = urls[1]
                k = hash(f"{src_url}\t{trg_url}")

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
                    logger.warning("Src or trg URL not in aligned URLs: %s\t%s", src_url, trg_url)

                    continue

                if ignore_duplicated_urls:
                    _skip = False

                    if src_url in src_urls_skipped:
                        logger.debug("Pair #%d: src URL ignored: %s", idx_aligned_urls, src_url)

                        _skip = True
                    if trg_url in trg_urls_skipped:
                        logger.debug("Pair #%d: trg URL ignored: %s", idx_aligned_urls, trg_url)

                        _skip = True

                    if _skip:
                        skipped_bc_duplicated += 1

                        continue

                score = 0.0
                nolines_score, occurrences_score = get_doc_nolines_score(src_url_nolines, trg_url_nolines, occurrences=occurrences,
                                                                         src_url=src_url, trg_url=trg_url)
                nolines_and_occurences_score_f1 = utils_bitextor.get_f1(nolines_score, occurrences_score)
                aligned_src_tokens = data["src_tokens"]
                aligned_trg_tokens = data["trg_tokens"]
                aligned_src_tokens_weighted_segalign = data["src_tokens_weighted_segalign"]
                aligned_trg_tokens_weighted_segalign = data["trg_tokens_weighted_segalign"]
                tokens_score = get_aligned_tokens_score(src_url_tokens, trg_url_tokens, aligned_src_tokens, aligned_trg_tokens,
                                                        src_url=src_url, trg_url=trg_url)
                tokens_score_weighted_segalign = get_aligned_tokens_score(src_url_tokens, trg_url_tokens, aligned_src_tokens_weighted_segalign,
                                                                          aligned_trg_tokens_weighted_segalign, check=False)

                if segalign_files_bicleaner_score_idx is not None:
                    aligned_src_tokens_weighted_bicleaner = data["src_tokens_weighted_bicleaner"]
                    aligned_trg_tokens_weighted_bicleaner = data["trg_tokens_weighted_bicleaner"]
                    aligned_src_tokens_weighted_segalign_and_bicleaner = data["src_tokens_weighted_segalign_and_bicleaner"]
                    aligned_trg_tokens_weighted_segalign_and_bicleaner = data["trg_tokens_weighted_segalign_and_bicleaner"]
                    tokens_score_weighted_bicleaner = get_aligned_tokens_score(src_url_tokens, trg_url_tokens, aligned_src_tokens_weighted_bicleaner,
                                                                               aligned_trg_tokens_weighted_bicleaner, check=False)
                    tokens_score_weighted_segalign_and_bicleaner = get_aligned_tokens_score(src_url_tokens, trg_url_tokens,
                                                                                            aligned_src_tokens_weighted_segalign_and_bicleaner * 0.5,
                                                                                            aligned_trg_tokens_weighted_segalign_and_bicleaner * 0.5,
                                                                                            check=False)

                try:
                    score = docalign_url_scores[k]
                except KeyError:
                    score = -1.0

                    logger.warning("Docalign score not found for URLs: ('%s', '%s')", src_url, trg_url)

                # We want to avoid scientific notation
                score = round(score, 4)
                nolines_score = round(nolines_score, 4)
                occurrences_score = round(occurrences_score, 4)
                nolines_and_occurences_score_f1 = round(nolines_and_occurences_score_f1, 4)
                tokens_score = round(tokens_score, 4)
                aligned_src_tokens_weighted_segalign = round(aligned_src_tokens_weighted_segalign, 4)
                aligned_trg_tokens_weighted_segalign = round(aligned_trg_tokens_weighted_segalign, 4)
                tokens_score_weighted_segalign = round(tokens_score_weighted_segalign, 4)

                sys.stdout.write(f"{src_url}\t{trg_url}\t{score}")
                sys.stdout.write(f"\t{src_url_nolines}\t{trg_url_nolines}\t{nolines_score}\t{src_url_tokens}\t{trg_url_tokens}")
                sys.stdout.write(f"\t{occurrences}\t{occurrences_score}\t{nolines_and_occurences_score_f1}")
                sys.stdout.write(f"\t{aligned_src_tokens}\t{aligned_trg_tokens}\t{tokens_score}")
                sys.stdout.write(f"\t{aligned_src_tokens_weighted_segalign}\t{aligned_trg_tokens_weighted_segalign}\t{tokens_score_weighted_segalign}")

                if segalign_files_bicleaner_score_idx is not None:
                    aligned_src_tokens_weighted_bicleaner = round(aligned_src_tokens_weighted_bicleaner, 4)
                    aligned_trg_tokens_weighted_bicleaner = round(aligned_trg_tokens_weighted_bicleaner, 4)
                    tokens_score_weighted_bicleaner = round(tokens_score_weighted_bicleaner, 4)
                    aligned_src_tokens_weighted_segalign_and_bicleaner = round(aligned_src_tokens_weighted_segalign_and_bicleaner, 4)
                    aligned_trg_tokens_weighted_segalign_and_bicleaner = round(aligned_trg_tokens_weighted_segalign_and_bicleaner, 4)
                    tokens_score_weighted_segalign_and_bicleaner = round(tokens_score_weighted_segalign_and_bicleaner, 4)

                    sys.stdout.write(f"\t{aligned_src_tokens_weighted_bicleaner}\t{aligned_trg_tokens_weighted_bicleaner}\t{tokens_score_weighted_bicleaner}")
                    sys.stdout.write(f"\t{aligned_src_tokens_weighted_segalign_and_bicleaner}\t{aligned_trg_tokens_weighted_segalign_and_bicleaner}")
                    sys.stdout.write(f"\t{tokens_score_weighted_segalign_and_bicleaner}")

                sys.stdout.write('\n')

                total_printed_urls += 1

            logger.info("Total skipped URLs (min occ., docalign, duplicated): (%d, %d, %d)",
                        skipped_bc_min_occ, skipped_bc_docalign, skipped_bc_duplicated)

            processed_segalign_files.add(segalign_file)

    logger.info("Total printed URLs: %d", total_printed_urls)

    diff_segalign_docalign_files = set(segalign_files).difference(processed_segalign_files)

    if len(diff_segalign_docalign_files) != 0:
        for diff_segalign_docalign_file in diff_segalign_docalign_files:
            logger.error("Segalign file not processed: %s", diff_segalign_docalign_file)

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Get aligned URLs")

    parser.add_argument('--docalign-mt-matches-files', nargs='+', help=".matches files")
    parser.add_argument('--src-url-files', nargs='+', required=True, help="Source url.gz files from sharding")
    parser.add_argument('--trg-url-files', nargs='+', required=True, help="Target url.gz files from sharding")
    parser.add_argument('--src-sentences-files', nargs='+', required=True, help="Source sentences.gz files from sharding")
    parser.add_argument('--trg-sentences-files', nargs='+', required=True, help="Target sentences.gz files from sharding")
    parser.add_argument('--src-sentences-preprocess-cmd',
                        help="Preprocess command to apply to the src sentences. "
                             "The provided command has to read sentences from stdin and print to stdout")
    parser.add_argument('--trg-sentences-preprocess-cmd',
                        help="Preprocess command to apply to the trg sentences. "
                             "The provided command has to read sentences from stdin and print to stdout")
    parser.add_argument('--segalign-files', nargs='*', help="Segalign files. If not provided, only docalign will be taken into account")
    parser.add_argument('--segalign-files-src-url-idx', type=int, default=0, help="Segalign files src URL idx")
    parser.add_argument('--segalign-files-trg-url-idx', type=int, default=1, help="Segalign files trg URL idx")
    parser.add_argument('--segalign-files-src-text-idx', type=int, default=2, help="Segalign files src text URL idx")
    parser.add_argument('--segalign-files-trg-text-idx', type=int, default=3, help="Segalign files trg text URL idx")
    parser.add_argument('--segalign-files-segalign-score-idx', type=int, default=4,
                        help="Segalign files segalign score idx. The expected score domain is [0, 1]")
    parser.add_argument('--segalign-files-bicleaner-score-idx', type=int, default=None,
                        help="Segalign files bicleaner score idx. The expected score domain is [0, 1]")
    parser.add_argument('--segalign-preprocess-cmd',
                        help="Preprocess command to apply to the src and trg alignments. "
                             "The provided command has to read pair of sentences separated by tab from stdin and print to stdout")
    parser.add_argument('--n-jobs', type=int, default=24,
                        help="Number of parallel jobs to use (-n means to use all CPUs - n + 1). If 0 is provided, parallelization is disabled")

    parser.add_argument('--ignore-duplicated-urls', action='store_true',
                        help="Ignore src and trg duplicated URLs. This should avoid errors with duplicated URLs when segalign files are "
                             "processed since when there are duplicated URLs, we can't be sure which document is the one which appears "
                             "in a specific segalign file (results might even be from all the duplicated URLs for a specific pair)")
    parser.add_argument('--min-occurrences', type=int, default=0, help="Min. occurrences of URLs pairs")
    parser.add_argument('--docalign-threshold', type=float, default=0.0, help="Docalign threshold")

    parser.add_argument('-q', '--quiet', action='store_true', help="Silent logging mode")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = initialization()

    logger = utils.set_up_logging_logger(logger, level=logging.INFO if args.quiet else logging.DEBUG)

    logger.debug("Arguments processed: {}".format(str(args)))

    main(args)
