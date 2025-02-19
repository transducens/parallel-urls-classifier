
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

import joblib
import numpy as np

_log_read_docs = 10000 # url.gz and sentences.gz files
_log_read_pairs = 10000 # segalign files
_tokenize = lambda s: tokenizer.tokenize(s, check_gaps=False, tokenizer="word_tokenize")
logger = logging.getLogger("parallel_urls_classifier")

def get_f1(score_1, score_2):
    return 2 * ((score_1 * score_2) / (score_1 + score_2)) if not np.isclose(score_1 + score_2, 0.0) else 0.0

# TODO replace get_f1 by get_harmonic_mean
def get_harmonic_mean(values):
  s = sum([1.0 / v if not (v == 0 or np.isclose(v, 0.0)) else 0.0 for v in values])

  return len(values) / s if not np.isclose(s, 0.0) else 0.0

def get_statistics_from_segalign(segalign_file, src_url_idx, trg_url_idx, src_text_idx, trg_text_idx, segalign_score_idx,
                                 preprocess_cmd=None, parallelize=True, n_jobs=1, segalign_files_bicleaner_score_idx=None):
    # Download NLTK model if not available
    utils.check_nltk_model("tokenizers/punkt", "punkt", download=True) # Download before parallel: https://github.com/nltk/nltk/issues/1576

    results = {}

    if preprocess_cmd:
        preprocess_cmd = shlex.split(preprocess_cmd)

    def update_ref(results, r):
        url = r["pair"]
        bicleaner_score = r["bicleaner_score"]

        try:
            results[url]["occurrences"] += 1
            results[url]["segalign_score"].append(r["segalign_score"])
            results[url]["src_tokens"] += r["len_src_tokens"]
            results[url]["trg_tokens"] += r["len_trg_tokens"]
            results[url]["src_tokens_weighted_segalign"] += r["len_src_tokens_weighted_segalign"]
            results[url]["trg_tokens_weighted_segalign"] += r["len_trg_tokens_weighted_segalign"]

            if bicleaner_score is not None:
                results[url]["bicleaner_score"].append(bicleaner_score)
                results[url]["src_tokens_weighted_bicleaner"] += r["len_src_tokens_weighted_bicleaner"]
                results[url]["trg_tokens_weighted_bicleaner"] += r["len_trg_tokens_weighted_bicleaner"]
                results[url]["src_tokens_weighted_segalign_and_bicleaner"] += r["len_src_tokens_weighted_segalign_and_bicleaner"]
                results[url]["trg_tokens_weighted_segalign_and_bicleaner"] += r["len_trg_tokens_weighted_segalign_and_bicleaner"]
        except KeyError:
            results[url] = {
                "occurrences": 1,
                "segalign_score": [r["segalign_score"]],
                "bicleaner_score": [bicleaner_score] if bicleaner_score is not None else [],
                "src_tokens": r["len_src_tokens"],
                "trg_tokens": r["len_trg_tokens"],
                "src_tokens_weighted_segalign": r["len_src_tokens_weighted_segalign"],
                "trg_tokens_weighted_segalign": r["len_trg_tokens_weighted_segalign"],
                "src_tokens_weighted_bicleaner": r["len_src_tokens_weighted_bicleaner"],
                "trg_tokens_weighted_bicleaner": r["len_trg_tokens_weighted_bicleaner"],
                "src_tokens_weighted_segalign_and_bicleaner": r["len_src_tokens_weighted_segalign_and_bicleaner"],
                "trg_tokens_weighted_segalign_and_bicleaner": r["len_trg_tokens_weighted_segalign_and_bicleaner"],
            }

    def process(idx, line, level, ref=None):
        logging.getLogger("parallel_urls_classifier").handlers = []
        logger = utils.set_up_logging_logger(logging.getLogger("parallel_urls_classifier"), level=level) # https://github.com/joblib/joblib/issues/1017

        line = line.split('\t')
        line[-1] = line[-1].rstrip('\n')
        src_url = line[src_url_idx]
        trg_url = line[trg_url_idx]
        src_text = line[src_text_idx]
        trg_text = line[trg_text_idx]
        segalign_score = float(line[segalign_score_idx]) # Expected: [0, 1]
        bicleaner_score = float(line[segalign_files_bicleaner_score_idx]) if segalign_files_bicleaner_score_idx is not None else None # Expected: [0, 1]
        url = f"{src_url}\t{trg_url}"
        pair = f"{src_text}\t{trg_text}"

        if preprocess_cmd:
            # Apply preprocess to src and trg text

            cmd = subprocess.Popen(preprocess_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            pair, err = cmd.communicate(pair.encode())
            rtn_code = cmd.returncode
            pair = pair.decode('utf-8', errors="backslashreplace").rstrip('\n')

            if rtn_code:
                logger.error("Preprocess cmd returning code != 0: %d", rtn_code)

            if err:
                logger.warning("Preprocess cmd printed content from stderr: %s",
                                err.decode('utf-8', errors="backslashreplace").rstrip('\n'))

        _pair = pair.split('\n')
        len_src_tokens = 0
        len_trg_tokens = 0

        if len(_pair) > 1:
            if preprocess_cmd:
                logger.warning("Pair #%d returned more than 1 entry: %d entries: this might be possible if Bifixer "
                               "was executed without --ignore_segmentation", idx, len(_pair))
            else:
                raise Exception(f"Pair #{idx} contains more than 1 entry: {len(_pair)} entries: bug?: {str(_pair)}")

        if segalign_score < 0.0 or segalign_score > 1.0:
            logger.error("Unexpected segalign score: %f not in [0, 1]: clipping value: %s", segalign_score, pair)

            segalign_score = max(min(segalign_score, 1.0), 0.0)

        for idx_entry, pair in enumerate(_pair):
            pair = pair.split('\t')

            if len(pair) != 2:
                raise Exception(f"Pair #{idx},{idx_entry} doesn't have 2 elements but {len(pair)}: {str(pair)}")

            len_src_tokens += len(_tokenize(pair[0].strip()))
            len_trg_tokens += len(_tokenize(pair[1].strip()))

        len_src_tokens_weighted_segalign = float(len_src_tokens) * segalign_score
        len_trg_tokens_weighted_segalign = float(len_trg_tokens) * segalign_score
        len_src_tokens_weighted_bicleaner = None
        len_trg_tokens_weighted_bicleaner = None
        len_src_tokens_weighted_segalign_and_bicleaner = None
        len_trg_tokens_weighted_segalign_and_bicleaner = None

        if bicleaner_score is not None:
            if bicleaner_score < 0.0 or bicleaner_score > 1.0:
                logger.error("Unexpected bicleaner score: %f not in [0, 1]: clipping value: %s", bicleaner_score, pair)

                bicleaner_score = max(min(bicleaner_score, 1.0), 0.0)

            len_src_tokens_weighted_bicleaner = float(len_src_tokens) * bicleaner_score
            len_trg_tokens_weighted_bicleaner = float(len_trg_tokens) * bicleaner_score

            # Use F1 of the previous segalign and bicleaner scores for weighting the src and trg tokens
            segalign_bicleaner_score_f1 = get_f1(segalign_score, bicleaner_score)
            len_src_tokens_weighted_segalign_and_bicleaner = float(len_src_tokens) * segalign_bicleaner_score_f1
            len_trg_tokens_weighted_segalign_and_bicleaner = float(len_trg_tokens) * segalign_bicleaner_score_f1

        if (idx % _log_read_pairs) == 0:
            logger.debug("Segalign file: pairs read: %d", idx)

        results = {
            "pair": url,
            "segalign_score": segalign_score,
            "bicleaner_score": bicleaner_score,
            "len_src_tokens": len_src_tokens,
            "len_trg_tokens": len_trg_tokens,
            "len_src_tokens_weighted_segalign": len_src_tokens_weighted_segalign,
            "len_trg_tokens_weighted_segalign": len_trg_tokens_weighted_segalign,
            "len_src_tokens_weighted_bicleaner": len_src_tokens_weighted_bicleaner,
            "len_trg_tokens_weighted_bicleaner": len_trg_tokens_weighted_bicleaner,
            "len_src_tokens_weighted_segalign_and_bicleaner": len_src_tokens_weighted_segalign_and_bicleaner,
            "len_trg_tokens_weighted_segalign_and_bicleaner": len_trg_tokens_weighted_segalign_and_bicleaner,
        }

        if ref is not None:
            update_ref(ref, results)

        return results

    if parallelize:
        if n_jobs == 0:
            logger.warning("Updating n_jobs: from %d to %d", n_jobs, 1)

            n_jobs = 1
        if n_jobs < 1:
            if n_jobs == -1:
                logger.warning("Using all CPUs")
            else:
                logger.warning("Using all CPUs minus %d", abs(n_jobs + 1))

    with utils.open_xz_or_gzip_or_plain(segalign_file) as fd:
        total_pairs_read = 0

        if parallelize:
            _results = \
                joblib.Parallel(n_jobs=n_jobs)( \
                joblib.delayed(process)(idx, line, logger.level) for idx, line in enumerate(fd, 1))

            total_pairs_read = len(_results)

            for r in _results:
                update_ref(results, r)
        else:
            for idx, line in enumerate(fd, 1):
                process(idx, line, logger.level, ref=results)

            total_pairs_read = idx

        logger.debug("Segalign file: total pairs read: %d", total_pairs_read)

    return results

def get_statistics_from_url_and_sentences(url_files, sentences_files, preprocess_cmd=None, n_jobs=1, parallelize=True,
                                          parallelize_files_instead=False):
    # Download NLTK model if not available
    utils.check_nltk_model("tokenizers/punkt", "punkt", download=True) # Download before parallel: https://github.com/nltk/nltk/issues/1576

    results = {}
    skipped = set()

    if preprocess_cmd:
        preprocess_cmd = shlex.split(preprocess_cmd)

    def process_document(idx, idx_fd, url_line, sentences_line, level, ref=None, ref_skipped=None):
        logging.getLogger("parallel_urls_classifier").handlers = []
        logger = utils.set_up_logging_logger(logging.getLogger("parallel_urls_classifier"), level=level) # https://github.com/joblib/joblib/issues/1017

        url_line = url_line.strip().replace('\t', ' ')
        sentences_line = sentences_line.strip()
        _results = {}
        _results[url_line] = {}
        _skipped = set()
        sentences_line = base64.b64decode(sentences_line).strip(b'\n').split(b'\n')
        sentences_line = b'\n'.join(map(lambda s: s.split(b'\t')[0], sentences_line)) # Get 1st column of text

        if preprocess_cmd:
            # Apply preprocess to text

            cmd = subprocess.Popen(preprocess_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            sentences_line, err = cmd.communicate(sentences_line)
            rtn_code = cmd.returncode

            if rtn_code:
                logger.error("Files url.gz and sentences.gz #%d,%d: preprocess cmd returning code != 0: %d", idx, idx_fd, rtn_code)

            if err:
                logger.warning("Files url.gz and sentences.gz #%d,%d: preprocess cmd printed content from stderr: %s",
                                idx, idx_fd, err.decode('utf-8', errors="backslashreplace").rstrip('\n'))

        sentences_line = sentences_line.decode('utf-8', errors="backslashreplace").strip().split('\n')

        # Get statistics

        _results[url_line]["nolines"] = len(sentences_line)
        _results[url_line]["tokens"] = sum([len(_tokenize(sentence.strip())) for sentence in sentences_line])

        if (idx_fd % _log_read_docs) == 0:
            logger.debug("Files url.gz and sentences.gz #%d: documents read: %d", idx, idx_fd)

        if ref is not None:
            if url_line in ref:
                # We don't check if _results[url_line] != ref[url_line] because the segalign might have multiple alignments of the same sentences
                logger.warning("Files url.gz and sentences.gz #%d,%d: URL already processed: skipping: %s",
                               idx, idx_fd, url_line)

                _skipped.add(url_line)

                if ref_skipped is not None:
                    ref_skipped.update(_skipped)
            else:
                ref[url_line] = _results[url_line]

        return _results, _skipped

    def process(idx, url_file, sentences_file, level, ref=None, ref_skipped=None):
        logging.getLogger("parallel_urls_classifier").handlers = []
        logger = utils.set_up_logging_logger(logging.getLogger("parallel_urls_classifier"), level=level) # https://github.com/joblib/joblib/issues/1017

        _results = {} if ref is None else ref
        _skipped = set() if ref_skipped is None else ref_skipped
        current_read_docs = 0

        with utils.open_xz_or_gzip_or_plain(url_file) as url_fd, utils.open_xz_or_gzip_or_plain(sentences_file) as sentences_fd:
            if parallelize and not parallelize_files_instead:
                _sentences_results = \
                    joblib.Parallel(n_jobs=n_jobs)( \
                    joblib.delayed(process_document)(idx, idx_fd, url_line, sentences_line, level) \
                        for idx_fd, (url_line, sentences_line) in enumerate(zip(url_fd, sentences_fd), 1))

                for r, s in _sentences_results:
                    _skipped.update(s)

                    if len(r.keys()) > 0:
                        if len(r.keys()) > 1:
                            raise Exception(f"Unexpected length: {len(r.keys())}")

                        k = list(r.keys())[0]

                        if k in _results:
                            # We don't check if r[k] != _results[k] because the segalign might have multiple alignments of the same sentences

                            logger.warning("Files url.gz and sentences.gz #%d: URL already processed: skipping: %s", idx, k)

                            _skipped.add(k)
                        else:
                            current_read_docs += len(r)
                            _results.update(r)
            else:
                for idx_fd, (url_line, sentences_line) in enumerate(zip(url_fd, sentences_fd), 1):
                    process_document(idx, idx_fd, url_line, sentences_line, level, ref=_results, ref_skipped=_skipped)

                current_read_docs += len(_results)

        logger.debug("Files url.gz and sentences.gz #%d: total documents read: %d", idx, current_read_docs)

        return _results, _skipped

    if parallelize:
        if n_jobs == 0:
            logger.warning("Updating n_jobs: from %d to %d", n_jobs, 1)

            n_jobs = 1
        if n_jobs < 1:
            if n_jobs == -1:
                logger.warning("Using all CPUs")
            else:
                logger.warning("Using all CPUs minus %d", abs(n_jobs + 1))

    if not parallelize or not parallelize_files_instead:
        for idx, (url_file, sentences_file) in enumerate(zip(url_files, sentences_files), 1):
            process(idx, url_file, sentences_file, logger.level, ref=results, ref_skipped=skipped)

            logger.debug("Files url.gz and sentences.gz #%d: unique documents accumulated: %d", idx, len(results))
    else:
        _results, _skipped = \
            joblib.Parallel(n_jobs=n_jobs)( \
            joblib.delayed(process)(idx, url_file, sentences_file, logger.level) \
                for idx, (url_file, sentences_file) in enumerate(zip(url_files, sentences_files), 1))

        for idx, (r, s) in enumerate(zip(_results, _skipped), 1):
            if results.keys().isdisjoint(r.keys()):
                logger.warning("Files url.gz and sentences.gz #%d: there are URLs which have already been processed: "
                               "results are going to be updated", idx)

            results.update(r)
            skipped.update(s)

            logger.debug("Files url.gz and sentences.gz #%d: unique documents accumulated: %d", idx, len(results))

        # Update skipped pairs with the combinations of the results
        for idx1 in range(len(_results)):
            for idx2 in range(idx1 + 1, len(_results)):
                skipped.update(set(_results[idx1].keys()).intersection(_results[idx2].keys()))

    return results, skipped

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
