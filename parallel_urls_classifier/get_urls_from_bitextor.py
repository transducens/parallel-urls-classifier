
import sys
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
            src_url = line[src_url_idx].replace('\t', ' ')
            trg_url = line[trg_url_idx].replace('\t', ' ')
            url = f"{src_url}\t{trg_url}"

            if url not in urls:
                urls[url] = 0

            urls[url] += 1
    
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

                urls_nolines[url_line] = sentences_line.count('\n') + 1

            logging.debug("Read url.gz and sentences.gz number of lines: %d", len(urls_nolines))

    return urls_nolines

def main(args):
    min_occurrences = args.min_occurrences
    sent_file = args.sent_file
    url_files = args.url_files
    sentences_files = args.sentences_files
    abs_sub_nolines = args.abs_sub_nolines

    if len(url_files) != len(sentences_files):
        raise Exception(f"Unexpected different number of files provided for url.gz and sentences.gz ({len(url_files)} vs {len(sentences_files)})")

    aligned_urls = get_urls_from_sent(sent_file, 0, 1)

    logging.info("Unique different paired URLs: %d", len(aligned_urls))

    urls_nolines = get_nolines_from_url_and_sentences(url_files, sentences_files)

    logging.info("Number of URLs: %d", len(urls_nolines))

    total_printed_urls = 0

    for idx, (url, occurrences) in enumerate(aligned_urls.items()):
        if (idx + 1) % 10000 == 0:
            #logging.debug("%.2f finished", (idx + 1) * 100.0 / len(aligned_urls))
            pass

        if occurrences < min_occurrences:
            continue

        urls = url.split('\t')

        assert len(urls) == 2, "Error"

        src_url = urls[0]
        trg_url = urls[1]

        try:
            src_url_nolines = urls_nolines[src_url]
            trg_url_nolines = urls_nolines[trg_url]
        except KeyError:
            logging.warning("src URL (%s) or trg URL (%s) not in aligned URLs", src_url, trg_url)

            continue

        max_url_nolines = max(src_url_nolines, trg_url_nolines)
        urls_diff = abs(src_url_nolines - trg_url_nolines)

        # Check out if the aligned URLs have very different number of lines
        if abs_sub_nolines >= 0:
            if urls_diff > abs_sub_nolines:
                continue
        elif max_url_nolines < 5:
            pass
        elif max_url_nolines < 10 and urls_diff > 2:
            continue
        elif max_url_nolines < 20 and urls_diff > 4:
            continue
        elif max_url_nolines < 40 and urls_diff > 6:
            continue
        elif urls_diff > 10:
            continue

        print(f"{src_url}\t{trg_url}")

        total_printed_urls += 1

    logging.info("Total printed URLs: %d", total_printed_urls)

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Get aligned URLs")

    parser.add_argument('sent_file', help=".sent.gz file")

    parser.add_argument('--url-files', nargs='+', required=True, help="url.gz files from sharding")
    parser.add_argument('--sentences-files', nargs='+', required=True, help="sentences.gz files from sharding")

    parser.add_argument('--min-occurrences', type=int, default=5, help="Min. occurrences of URLs pairs")
    parser.add_argument('--abs-sub-nolines', type=int, default=-1,
                        help="By default, the number of lines of the documents are handled taking into account the "
                             "relative size of the documents. If this options is set, the provided number will be "
                             "the max. number of lines which a pair of documents can have when subtracting the number "
                             "of lines")

    parser.add_argument('-q', '--quiet', action='store_true', help="Silent logging mode")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = initialization()

    utils.set_up_logging(level=logging.INFO if args.quiet else logging.DEBUG)

    logging.debug("Arguments processed: {}".format(str(args)))

    main(args)
