
import math
import base64
import logging
import argparse

import utils

def get_urls_from_sent(sent_file, src_url_idx, trg_url_idx):
    urls = {}

    with utils.open_xz_or_gzip_or_plain(sent_file) as fd:
        for line in fd:
            line = line.split('\t')
            line[-1] = line[-1].rstrip('\n')
            src_url = line[src_url_idx]
            trg_url = line[trg_url_idx]
            url = f"{src_url}\t{trg_url}"

            try:
                urls[url] += 1
            except KeyError:
                urls[url] = 1

    return urls

def get_nolines_from_url_and_sentences(url_files, sentences_files):
    urls_nolines = {}

    for url_file, sentences_file in zip(url_files, sentences_files):
        with utils.open_xz_or_gzip_or_plain(url_file) as url_fd, utils.open_xz_or_gzip_or_plain(sentences_file) as sentences_fd:
            for url_line, sentences_line in zip(url_fd, sentences_fd):
                url_line = url_line.strip().replace('\t', '')
                sentences_line = sentences_line.strip()

                # URL should not be the same twice
                sentences_line = base64.b64decode(sentences_line).decode('utf-8', errors="ignore").strip()

                urls_nolines[url_line] = sentences_line.count('\n') + (1 if sentences_line != '' else 0)

            logging.debug("Read url.gz and sentences.gz number of lines: %d", len(urls_nolines))

    return urls_nolines

def get_urls_from_idxs(url_file, idxs):
    idxs_sorted = sorted(idxs)
    idxs_dict = {}
    urls = []
    current_idx_idx = 0

    with utils.open_xz_or_gzip_or_plain(url_file) as f:
        for idx, url in enumerate(f, 1):
            if current_idx_idx >= len(idxs_sorted):
                break

            if idx < idxs_sorted[current_idx_idx]:
                continue

            url = url.strip().replace('\t', ' ')
            idxs_dict[idx] = url

            current_idx_idx += 1

    for idx in idxs:
        urls.append(idxs_dict[idx])

    assert current_idx_idx == len(idxs_sorted), f"current_idx_idx != len(idxs_sorted) = {current_idx_idx} != {len(idxs_sorted)}"
    assert len(idxs_sorted) == len(urls), f"len(idxs_sorted) != len(urls) = {len(idxs_sorted)} != {len(urls)}"

    return urls

def main(args):
    min_occurrences = args.min_occurrences
    sent_file = args.sent_file
    docalign_mt_matches_files = args.docalign_mt_matches_files
    src_url_files = args.src_url_files
    trg_url_files = args.trg_url_files
    src_sentences_files = args.src_sentences_files
    trg_sentences_files = args.trg_sentences_files
    abs_sub_nolines = args.abs_sub_nolines
    docalign_threshold = args.docalign_threshold
    print_docalign_score = args.print_docalign_score

    if len(src_url_files) != len(src_sentences_files):
        raise Exception("Unexpected different number of source files provided for url.gz and sentences.gz " \
                        f"({len(src_url_files)} vs {len(src_sentences_files)})")
    if len(trg_url_files) != len(trg_sentences_files):
        raise Exception("Unexpected different number of target files provided for url.gz and sentences.gz " \
                        f"({len(trg_url_files)} vs {len(trg_sentences_files)})")

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

    for shard in shards:
        urls_quantity = len(urls_shard_batch["src"][shard]) * len(urls_shard_batch["trg"][shard])

        if urls_quantity != docalign_mt_matches_shard_quantity[shard]:
            raise Exception("Unexpected quantity of url.gz/sentences.gz files taking into account matches files " \
                            f"(shard {shard}: {urls_quantity} vs {docalign_mt_matches_shard_quantity[shard]})")

    docalign_src_urls, docalign_trg_urls = {}, {}
    docalign_url_scores = {}
    total_printed_urls = 0

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

        src_urls = get_urls_from_idxs(urls_shard_batch["src"][docalign_mt_matches_shard_batch[0]][docalign_mt_matches_shard_batch[1][0]], src_urls_idx)
        trg_urls = get_urls_from_idxs(urls_shard_batch["trg"][docalign_mt_matches_shard_batch[0]][docalign_mt_matches_shard_batch[1][1]], trg_urls_idx)

        if len(src_urls) != len(trg_urls):
            raise Exception("Unexpected different number of src and trg URLs "
                            f"({len(src_urls)} vs {len(trg_urls)}) in file '{docalign_mt_matches_file}'")
        if len(src_urls) != len(scores):
            raise Exception("Unexpected different number of URLs and scores "
                            f"({len(src_urls)} vs {len(scores)}) in file '{docalign_mt_matches_file}'")

        logging.debug("Number of aligned URLs from '%s': %d", docalign_mt_matches_file, len(src_urls))

        for src_url, trg_url in zip(src_urls, trg_urls):
            try:
                docalign_src_urls[src_url] += 1
            except KeyError:
                docalign_src_urls[src_url] = 1
            try:
                docalign_trg_urls[trg_url] += 1
            except KeyError:
                docalign_trg_urls[trg_url] = 1

        for score, src_url, trg_url in zip(scores, src_urls, trg_urls):
            k = hash(f"{src_url}\t{trg_url}")
            docalign_url_scores[k] = score

            if not sent_file:
                if print_docalign_score:
                    print(f"{score}\t{src_url}\t{trg_url}")
                else:
                    print(f"{src_url}\t{trg_url}")

                total_printed_urls += 1

    # Process sent.gz file
    if sent_file:
        logging.info("Processing sent.gz file")

        aligned_urls = get_urls_from_sent(sent_file, 0, 1)

        logging.info("Unique different paired URLs: %d", len(aligned_urls))

        src_urls_nolines = get_nolines_from_url_and_sentences(src_url_files, src_sentences_files)
        trg_urls_nolines = get_nolines_from_url_and_sentences(trg_url_files, trg_sentences_files)

        logging.info("Number of URLs (src, trg): (%d, %d)", len(src_urls_nolines), len(trg_urls_nolines))

        skipped_bc_docalign = 0
        skipped_bc_url_diff = 0
        skipped_bc_min_occ = 0

        for idx, (url, occurrences) in enumerate(aligned_urls.items()):
            if (idx + 1) % 10000 == 0:
                logging.debug("%.2f finished", (idx + 1) * 100.0 / len(aligned_urls))
                logging.debug("Currently skipped URLs (min occ., docalign, diff. nolines): (%d, %d, %d)",
                              skipped_bc_min_occ, skipped_bc_docalign, skipped_bc_url_diff)

            if occurrences < min_occurrences:
                skipped_bc_min_occ += 1
                continue

            urls = url.split('\t')

            assert len(urls) == 2, "Error"

            src_url = urls[0]
            trg_url = urls[1]

            try:
                docalign_src_urls[src_url]
                docalign_trg_urls[trg_url]
            except KeyError:
                skipped_bc_docalign += 1
                continue

            try:
                src_url_nolines = src_urls_nolines[src_url]
                trg_url_nolines = trg_urls_nolines[trg_url]
            except KeyError:
                logging.warning("src URL (%s) or trg URL (%s) not in aligned URLs", src_url, trg_url)

                continue

            max_url_nolines = max(src_url_nolines, trg_url_nolines)
            urls_diff = abs(src_url_nolines - trg_url_nolines)

            # Check out if the aligned URLs have very different number of lines
            if abs_sub_nolines >= 0:
                if urls_diff > abs_sub_nolines:
                    skipped_bc_url_diff += 1
                    continue
            elif max_url_nolines >= 5:
                actual_urls_diff = math.log2(2 * max_url_nolines - 2 * 8) # The equation is very close to realistic values
                actual_urls_diff = round(actual_urls_diff)

                if urls_diff > actual_urls_diff:
                    skipped_bc_url_diff += 1
                    continue

            if print_docalign_score:
                k = hash(f"{src_url}\t{trg_url}")
                score = 0.0

                try:
                    score = docalign_url_scores[k]
                except KeyError:
                    score = -1.0

                    logging.warning("Score not found for URLs: ('%s', '%s')", src_url, trg_url)

                print(f"{score}\t{src_url}\t{trg_url}")
            else:
                print(f"{src_url}\t{trg_url}")

            total_printed_urls += 1

    logging.info("Total skipped URLs (min occ., docalign, diff. nolines): (%d, %d, %d)",
                 skipped_bc_min_occ, skipped_bc_docalign, skipped_bc_url_diff)
    logging.info("Total printed URLs: %d", total_printed_urls)

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Get aligned URLs")

    parser.add_argument('--docalign-mt-matches-files', nargs='+', required=True, help=".matches files")
    parser.add_argument('--src-url-files', nargs='+', required=True, help="Source url.gz files from sharding")
    parser.add_argument('--trg-url-files', nargs='+', required=True, help="Target url.gz files from sharding")
    parser.add_argument('--src-sentences-files', nargs='+', required=True, help="Source sentences.gz files from sharding")
    parser.add_argument('--trg-sentences-files', nargs='+', required=True, help="Target sentences.gz files from sharding")
    parser.add_argument('--sent-file', help=".sent.gz file. If not provided, only docalign will be taken into account")

    parser.add_argument('--min-occurrences', type=int, default=5, help="Min. occurrences of URLs pairs")
    parser.add_argument('--abs-sub-nolines', type=int, default=-1,
                        help="By default, the number of lines of the documents are handled taking into account the "
                             "relative size of the documents. If this options is set, the provided number will be "
                             "the max. number of lines which a pair of documents can have when subtracting the number "
                             "of lines")
    parser.add_argument('--docalign-threshold', type=float, default=0.1, help="Docalign threshold")
    parser.add_argument('--print-docalign-score', action='store_true', help="Print docalign threshold")

    parser.add_argument('-q', '--quiet', action='store_true', help="Silent logging mode")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = initialization()

    utils.set_up_logging(level=logging.INFO if args.quiet else logging.DEBUG)

    logging.debug("Arguments processed: {}".format(str(args)))

    main(args)
