
import os
import sys
import base64
import logging

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, f"{cdir}/../..")

import parallel_urls_classifier.utils.utils as utils

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
                url_line = url_line.strip().replace('\t', ' ')
                sentences_line = sentences_line.strip()

                # URL should not be the same twice
                sentences_line = base64.b64decode(sentences_line).decode('utf-8', errors="backslashreplace").strip()

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
