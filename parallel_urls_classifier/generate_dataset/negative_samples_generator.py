
import re
import os
import sys
import random
import logging
import itertools
import copy

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, f"{cdir}/../..")

from parallel_urls_classifier.tokenizer import tokenize
from parallel_urls_classifier.generate_dataset.word_freqs_double_linked import WordFreqDistDoubleLinked
import parallel_urls_classifier.utils.utils as utils

import joblib

def get_negative_samples_replace_freq_words(parallel_urls, limit_max_alignments_per_url=10, min_replacements=1,
                                            src_monolingual_file='', trg_monolingual_file='', side="both",
                                            n_jobs=1):
    """
    Original function from https://github.com/bitextor/bicleaner-ai/blob/master/bicleaner_ai/training.py
    """
    if side not in ("src", "trg", "both", "all"):
        raise Exception(f"'side' must be in ('src', 'trg', 'both'): '{side}'")

    if side == "src" and not utils.exists(src_monolingual_file):
        raise Exception(f"Src monolingual file does not exist: '{src_monolingual_file}'")
    elif side == "trg" and not utils.exists(trg_monolingual_file):
        raise Exception(f"Trg monolingual file does not exist: '{trg_monolingual_file}'")
    elif side in ("both", "all") and (not utils.exists(src_monolingual_file) or not utils.exists(trg_monolingual_file)):
        raise Exception(f"Either src, trg or both monolingual files do not exist: ('{src_monolingual_file}', '{trg_monolingual_file}')")

    if side in ("src", "both", "all"):
        double_linked_freqs_src = WordFreqDistDoubleLinked(src_monolingual_file)
    if side in ("trg", "both", "all"):
        double_linked_freqs_trg = WordFreqDistDoubleLinked(trg_monolingual_file)

    urls = set()

    def run(tokenized_src_url, tokenized_trg_url, side="both", max_tries=6, percentage_tokens_affected=0.5, min_replacements=1):
        if side not in ("src", "trg", "both"):
            raise Exception(f"'side' must be in ('src', 'trg', 'both'): '{side}'")

        # Get URLs which should be processed
        tokenized_url_and_dic = []
        orig_src_url = copy.copy(tokenized_src_url)
        orig_trg_url = copy.copy(tokenized_trg_url)

        if side == "src":
            tokenized_url_and_dic = [(tokenized_src_url, double_linked_freqs_src)]
        elif side == "trg":
            tokenized_url_and_dic = [(tokenized_trg_url, double_linked_freqs_trg)]
        elif side == "both":
            tokenized_url_and_dic = [(tokenized_src_url, double_linked_freqs_src),
                                     (tokenized_trg_url, double_linked_freqs_trg)]

        # Replace tokens with equal frequency
        for tokenized_url, double_linked_freqs in tokenized_url_and_dic:
            # sentence will modify either tokenized_src_url, tokenized_trg_url or both (i.e. is a reference)

            count = 0

            # Loop until any of the chosen words have an alternative, at most 'max_tries' times
            while True:
                # Random number of words that will be replaced (at least 1)
                idx_words_to_replace = get_position_idxs_random(tokenized_url, percentage=percentage_tokens_affected, min_elements=1)
                total_replacements = 0

                for wordpos in idx_words_to_replace:
                    w = tokenized_url[wordpos] # lower() will be applied by WordFreqDistDoubleLinked instance
                    wfreq = double_linked_freqs.get_word_freq(w)
                    alternatives = double_linked_freqs.get_words_for_freq(wfreq)

                    # Avoid replace with the same word
                    if alternatives and w.lower() in alternatives:
                        alternatives.remove(w.lower())

                    # Try a non-exact approach, if needed
                    if not alternatives:
                        # Let's try a non-exact approach
                        woccs = double_linked_freqs.get_word_occs(w)
                        alternatives = double_linked_freqs.get_words_for_occs(woccs, exact=False)

                        # Avoid replace with the same word
                        if alternatives and w.lower() in alternatives:
                            alternatives.remove(w.lower())

                    if alternatives:
                        alternatives = list(alternatives)
                        tokenized_url[wordpos] = random.choice(alternatives)

                        # Restore starting capital letter
                        if wordpos == 0 and w[0].isupper():
                            tokenized_url[wordpos] = tokenized_url[wordpos].capitalize()

                        total_replacements += 1

                if total_replacements < min_replacements:
                    tokenized_src_url = orig_src_url
                    tokenized_trg_url = orig_trg_url

                count += 1

                if tokenized_src_url != orig_src_url or tokenized_trg_url != orig_trg_url:
                    # We got different URLs! 1 hit, at least (we might have had more since 'len(tokenized_url_and_dic) > 1' is possible)
                    # We can't restore original URLs because we might remove the modifications done by
                    #  other loop instance from 'tokenized_url_and_dic' element
                    break
                if count >= max_tries:
                    break

        return tokenized_src_url, tokenized_trg_url

    side_priority = ("trg", "src", "both") # Priority for modifying src, trg or both URLs -> the priority is important since it will depend on 'limit_max_alignments_per_url'

    def process(src_url, trg_url, idx):
        hit = False
        _side = side

        if side == "all":
            _side = side_priority[idx % len(side_priority)] # Take a side taking into account the specified priority

        _src_url, _trg_url = apply_function_to_negative_sample_tokenized_urls(
            src_url, trg_url,
            lambda s, t: run(s, t, side=_side, min_replacements=min_replacements),
            "replace_freq_words")

        if src_url != _src_url or trg_url != _trg_url:
            hit = True

            return _src_url, _trg_url, hit, idx

        return src_url, trg_url, hit, f"{side}/{_side}"

    _results = \
        joblib.Parallel(n_jobs=n_jobs)( \
        joblib.delayed(process)(src_url, trg_url, idx) for src_url, trg_url in parallel_urls for idx in range(limit_max_alignments_per_url))

    for _src_url, _trg_url, hit, side_strategy in _results:
        if hit:
            urls.add((_src_url, _trg_url))
        else:
            logging.warning("Couldn't find words with the same frequency: couldn't generate negative sample for the provided URLs with the side '%s': ('%s', '%s')",
                            side_strategy, str(_src_url), str(_trg_url))

    common_last_checks(urls, parallel_urls)

    return list(urls)

def get_negative_samples_remove_random_tokens(parallel_urls, limit_max_alignments_per_url=10, remove_percentage=0.5, n_jobs=1):
    if remove_percentage < 0.0 or remove_percentage > 1.0:
        raise Exception(f"0.0 <= remove_percentage <= 1.0: {remove_percentage}")

    urls = set()

    def run(tokenized_src_url, tokenized_trg_url):
        # Remove elements (at least 1)
        src_url_remove_idxs = get_position_idxs_random(tokenized_src_url, percentage=remove_percentage, min_elements=1)
        trg_url_remove_idxs = get_position_idxs_random(tokenized_trg_url, percentage=remove_percentage, min_elements=1)

        if not src_url_remove_idxs and not trg_url_remove_idxs:
            logging.warning("Couldn't get indexes to remove for the provided src and trg URLs: ('%s', '%s')",
                            str(tokenized_src_url), str(tokenized_trg_url))

        for idx in src_url_remove_idxs:
            tokenized_src_url.pop(idx)
        for idx in trg_url_remove_idxs:
            tokenized_trg_url.pop(idx)

        return tokenized_src_url, tokenized_trg_url

    _results = \
        joblib.Parallel(n_jobs=n_jobs)( \
        joblib.delayed(apply_function_to_negative_sample_tokenized_urls)(src_url, trg_url, run, "remove_random_tokens") \
            for src_url, trg_url in parallel_urls for _ in range(limit_max_alignments_per_url))

    for _src_url, _trg_url in _results:
        urls.add((_src_url, _trg_url))

    common_last_checks(urls, parallel_urls)

    return list(urls)

def get_negative_samples_intersection_metric(parallel_urls, limit_max_alignments_per_url=10, append_metric=False, n_jobs=1,
                                             apply_resource_forward=True):
    # Download NLTK model if not available
    utils.check_nltk_model("tokenizers/punkt", "punkt", download=True) # Download before parallel: https://github.com/nltk/nltk/issues/1576

    parallel_urls_dict = {}
    urls = set()

    def tokenize_urls(src_url, trg_url):
        resource_idx_src_url = utils.get_idx_resource(src_url) if apply_resource_forward else 0
        resource_idx_trg_url = utils.get_idx_resource(trg_url) if apply_resource_forward else 0
        tokenized_src_url = tokenize(src_url[resource_idx_src_url:])
        tokenized_trg_url = tokenize(trg_url[resource_idx_trg_url:])

        return tokenized_src_url, tokenized_trg_url

    tokenized_urls = \
        joblib.Parallel(n_jobs=n_jobs)( \
        joblib.delayed(tokenize_urls)(src_url, trg_url) for src_url, trg_url in parallel_urls)

    def get_metrics(src_url, trg_url, idx_pair_src_url, idx_pair_trg_url):
        tokenized_src_url = tokenized_urls[idx_pair_src_url][0]
        tokenized_trg_url = tokenized_urls[idx_pair_trg_url][1]
        metric1 = len(set(tokenized_src_url).intersection(set(tokenized_trg_url)))
        metric2 = len(set(src_url).intersection(set(trg_url)))

        return trg_url, (metric1, metric2)

    for idx_pair_src_url, parallel_urls_pair in enumerate(parallel_urls):
        src_url, _ = parallel_urls_pair
        _results = \
            joblib.Parallel(n_jobs=n_jobs)( \
            joblib.delayed(get_metrics)(src_url, trg_url, idx_pair_src_url, idx_pair_trg_url) \
                for idx_pair_trg_url, (_, trg_url) in enumerate(parallel_urls) if idx_pair_src_url != idx_pair_trg_url)
        sorted_trg_parallel_urls_dict = sorted(_results, key=lambda item: (item[1][0], item[1][1]), reverse=True)

        for idx, (trg_url, metrics) in enumerate(sorted_trg_parallel_urls_dict):
            if idx >= limit_max_alignments_per_url:
                break

            if append_metric:
                urls.add((src_url, trg_url, *metrics))
            else:
                urls.add((src_url, trg_url))

    common_last_checks(urls, parallel_urls)

    return list(urls)

def get_negative_samples_random(parallel_urls, limit_max_alignments_per_url=10, n_jobs=1):
    #idxs = range(len(parallel_urls))
    urls = set()

    def get_random_trg_urls(idx1):
        _urls = set()
        max_alignments_per_url = limit_max_alignments_per_url
        sample_idxs = random.sample(range(len(parallel_urls)), limit_max_alignments_per_url) # https://joblib.readthedocs.io/en/latest/auto_examples/parallel_random_state.html

        for sort_idx2, idx2 in enumerate(sample_idxs, 1):
            if idx1 == idx2:
                # Skip parallel URLs
                max_alignments_per_url += 1

                continue

            if sort_idx2 > max_alignments_per_url:
                break

            src_pair = parallel_urls[idx1]
            trg_pair = parallel_urls[idx2]
            src_url = src_pair[0]
            trg_url = trg_pair[1]

            _urls.add((src_url, trg_url))

        return _urls

    _results = \
        joblib.Parallel(n_jobs=n_jobs)( \
        joblib.delayed(get_random_trg_urls)(idx) for idx in range(len(parallel_urls)))

    for _r in _results:
        urls.update(_r)

    common_last_checks(urls, parallel_urls)

    return list(urls)

_long_warning_raised = False

def common_last_checks(negative_samples_set, parallel_urls_set):
    global _long_warning_raised

    urls_len = len(negative_samples_set)

    # Update reference set removing parallel pairs, if any
    negative_samples_set.difference_update(parallel_urls_set)

    urls_overlap = urls_len - len(negative_samples_set)

    if urls_overlap > 0:
        if _long_warning_raised:
            logging.warning("Parallel and non-parallel URLs sets overlap > 0: %d", urls_overlap)
        else:
            logging.warning("Parallel and non-parallel URLs sets overlap > 0: "
                            "this might happen for different reasons (e.g. your strategy couldn't generate a negative sample, "
                            "you have provided >1 pair of URLs where are >1 translation for the same document): %d", urls_overlap)

            _long_warning_raised = True

def show_info_from_fd(fd, generator, generator_kwargs=None, sample_size=None, print_size=None, print_results=True):
    urls = []

    for idx, url_pair in enumerate(fd):
        if sample_size is not None and idx >= sample_size:
            break

        url_pair = url_pair.strip().split('\t')

        assert len(url_pair) == 2, f"The provided line does not have 2 tab-separated values (line #{idx + 1})"

        urls.append((url_pair[0], url_pair[1]))

    if generator_kwargs:
        results = generator(urls, **generator_kwargs)
    else:
        results = generator(urls)

    if print_results:
        for idx, r in enumerate(results):
            if print_size is not None and idx >= print_size:
                break

            print(str(r))

    return results

def apply_function_to_negative_sample_tokenized_urls(src_url, trg_url, func, strategy_description, apply_resource_forward=True):
    """
    Input is str URLs
    Input of 'func' is src and trg tokenized URLs
    Output of 'func' must be src and trg tuple of lists or strings
    Output is str URLs
    """
    h = hash(f"{src_url}\t{trg_url}")

    # Get tokens
    resource_idx_src_url = utils.get_idx_resource(src_url) if apply_resource_forward else 0
    resource_idx_trg_url = utils.get_idx_resource(trg_url) if apply_resource_forward else 0
    tokenized_src_url = tokenize(src_url[resource_idx_src_url:])
    tokenized_trg_url = tokenize(trg_url[resource_idx_trg_url:])

    # We want to apply the function to URL resource forward
    src_url_before_resource = src_url[:resource_idx_src_url]
    trg_url_before_resource = trg_url[:resource_idx_trg_url]

    # Apply function which will modify the tokens
    tokenized_src_url, tokenized_trg_url = func(tokenized_src_url, tokenized_trg_url)

    # Reconstruct
    src_url_again = src_url_before_resource + (''.join(tokenized_src_url) if isinstance(tokenized_src_url, list) else tokenized_src_url)
    trg_url_again = trg_url_before_resource + (''.join(tokenized_trg_url) if isinstance(tokenized_trg_url, list) else tokenized_trg_url)

    if hash(f"{src_url_again}\t{trg_url_again}") == h:
        # Do not add a negative sample which actually is a positive sample
        logging.warning("Couldn't generate a negative sample using '%s': return original src and trg URLs: ('%s', '%s')",
                        strategy_description, src_url, trg_url)

        return src_url, trg_url

    # Replace multiple '/' with one '/' (only the protocol has to have 2 '/')
    protocol_idx_src_url = utils.get_idx_after_protocol(src_url_again)
    protocol_idx_trg_url = utils.get_idx_after_protocol(trg_url_again)
    src_url_again = src_url_again[:protocol_idx_src_url] + re.sub(r'/+', r'/', src_url_again[protocol_idx_src_url:])
    trg_url_again = trg_url_again[:protocol_idx_trg_url] + re.sub(r'/+', r'/', trg_url_again[protocol_idx_trg_url:])

    return src_url_again, trg_url_again

def get_position_idxs_random(l, percentage=0.5, min_elements=1, reverse=True):
    if percentage < 0.0:
        percentage = 0.0

        logging.warning("Percentage set to 0.0")
    elif percentage > 1.0:
        percentage = 1.0

        logging.warning("Percentage set to 1.0")

    k = random.normalvariate(mu=len(l) * percentage, sigma=len(l) * percentage / 3.0) # 99.8 % likelihood of obtaining in-domain index in relation with the provided percentage and length of l
                                                                                      # https://en.wikipedia.org/wiki/Normal_distribution#/media/File:Standard_deviation_diagram_micro.svg
    k = min(max(round(k), min_elements), len(l))
    idxs = sorted(random.sample(range(len(l)), k), reverse=reverse)

    return idxs
