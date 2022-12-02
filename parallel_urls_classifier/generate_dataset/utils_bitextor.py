
import os
import sys
import base64
import logging
import subprocess
import shlex

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, f"{cdir}/../..")

import parallel_urls_classifier.utils.utils as utils
import parallel_urls_classifier.tokenizer as tokenizer

_tokenize = lambda s: tokenizer.tokenize(s, check_gaps=False, tokenizer="word_tokenize")

def get_statistics_from_raw(raw_file, src_url_idx, trg_url_idx, src_text_idx, trg_text_idx, bicleaner_idx=None, preprocess_cmd=None):
    results = {}

    if preprocess_cmd:
        preprocess_cmd = shlex.split(preprocess_cmd)

    with utils.open_xz_or_gzip_or_plain(raw_file) as fd:
        for line in fd:
            line = line.split('\t')
            line[-1] = line[-1].rstrip('\n')
            src_url = line[src_url_idx]
            trg_url = line[trg_url_idx]
            src_text = line[src_text_idx]
            trg_text = line[trg_text_idx]
            bicleaner = float(line[bicleaner_idx]) if bicleaner_idx is not None else -1.0
            url = f"{src_url}\t{trg_url}"
            pair = f"{src_text}\t{trg_text}"

            if preprocess_cmd:
                # Apply preprocess to src and trg text

                cmd = subprocess.Popen(preprocess_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                pair, err = cmd.communicate(pair.encode())
                rtn_code = cmd.return_code
                pair = pair.decode('utf-8', errors="backslashreplace").rstrip('\n')

                if not rtn_code:
                    logging.error("Preprocess cmd returning code != 0: %d", rtn_code)

                if err:
                    logging.warning("Preprocess cmd printed content from stderr: %s",
                                    err.decode('utf-8', errors="backslashreplace").rstrip('\n'))

            pair = pair.split('\t')

            if len(pair) != 2:
                raise Exception(f"Pair doesn't have 2 elements but {len(pair)}: pair: {str(pair)}")

            len_src_tokens = len(_tokenize(pair[0].strip()))
            len_trg_tokens = len(_tokenize(pair[1].strip()))

            try:
                if bicleaner_idx is not None:
                    results[url]["bicleaner_sum"] += bicleaner

                results[url]["occurrences"] += 1
                results[url]["src_tokens"] += len_src_tokens
                results[url]["trg_tokens"] += len_trg_tokens
            except KeyError:
                results[url] = {
                    "occurrences": 1,
                    "bicleaner_sum": bicleaner,
                    "src_tokens": len_src_tokens,
                    "trg_tokens": len_trg_tokens,
                }

    return results

def get_statistics_from_url_and_sentences(url_files, sentences_files, preprocess_cmd=None):
    results = {}

    if preprocess_cmd:
        preprocess_cmd = shlex.split(preprocess_cmd)

    for url_file, sentences_file in zip(url_files, sentences_files):
        with utils.open_xz_or_gzip_or_plain(url_file) as url_fd, utils.open_xz_or_gzip_or_plain(sentences_file) as sentences_fd:
            for url_line, sentences_line in zip(url_fd, sentences_fd):
                url_line = url_line.strip().replace('\t', ' ')
                sentences_line = sentences_line.strip()

                # URL should not be the same twice
                sentences_line = base64.b64decode(sentences_line).strip() #.decode('utf-8', errors="backslashreplace").strip() #.split('\n')

                if preprocess_cmd:
                    # Apply preprocess to text

                    cmd = subprocess.Popen(preprocess_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    sentences_line, err = cmd.communicate(sentences_line)
                    rtn_code = cmd.return_code

                    if not rtn_code:
                        logging.error("Preprocess cmd returning code != 0: %d", rtn_code)

                    if err:
                        logging.warning("Preprocess cmd printed content from stderr: %s",
                                        err.decode('utf-8', errors="backslashreplace").rstrip('\n'))

                sentences_line = sentences_line.decode('utf-8', errors="backslashreplace").strip().split('\n')

                # Get statistics
                if url_line in results:
                    logging.warning("URL already processed: previous result is going to be replaced")
                else:
                    results[url_line] = {}

                results[url_line]["nolines"] = len(sentences_line)
                results[url_line]["tokens"] = [len(_tokenize(sentence.split())) for sentence in sentences_line]

            logging.debug("Read url.gz and sentences.gz number of lines: %d", len(results))

    return results

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
