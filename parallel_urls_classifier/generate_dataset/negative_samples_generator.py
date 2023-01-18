
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
from tldextract import extract

_double_linked_freqs_src = None
_double_linked_freqs_trg = None
def get_negative_samples_replace_freq_words(parallel_urls, limit_max_alignments_per_url=10, min_replacements=1,
                                            src_monolingual_file='', trg_monolingual_file='', side="both",
                                            n_jobs=1):
    """
    Original function from https://github.com/bitextor/bicleaner-ai/blob/master/bicleaner_ai/training.py
    """
    if side not in ("src", "trg", "both", "all", "all-any"):
        raise Exception(f"'side' must be in ('src', 'trg', 'both', 'all', 'all-any'): '{side}'")

    if side == "src" and not utils.exists(src_monolingual_file):
        raise Exception(f"Src monolingual file does not exist: '{src_monolingual_file}'")
    elif side == "trg" and not utils.exists(trg_monolingual_file):
        raise Exception(f"Trg monolingual file does not exist: '{trg_monolingual_file}'")
    elif side in ("both", "all", "all-any") and (not utils.exists(src_monolingual_file) or not utils.exists(trg_monolingual_file)):
        raise Exception(f"Either src, trg or both monolingual files do not exist: ('{src_monolingual_file}', '{trg_monolingual_file}')")

    global _double_linked_freqs_src
    global _double_linked_freqs_trg

    if side in ("src", "both", "all", "all-any"):
        if _double_linked_freqs_src is None:
            logging.debug("Loading src word freq.: %s", trg_monolingual_file)

            _double_linked_freqs_src = WordFreqDistDoubleLinked(src_monolingual_file)

            logging.debug("Src word freq. loaded")

        double_linked_freqs_src = _double_linked_freqs_src
    if side in ("trg", "both", "all", "all-any"):
        if _double_linked_freqs_trg is None:
            logging.debug("Loading trg word freq.: %s", trg_monolingual_file)

            _double_linked_freqs_trg = WordFreqDistDoubleLinked(trg_monolingual_file)

            logging.debug("Trg word freq. loaded")

        double_linked_freqs_trg = _double_linked_freqs_trg

    urls = set()

    def run(tokenized_src_url, tokenized_trg_url, side="both", max_tries=6, percentage_tokens_affected=0.5, min_replacements=1):
        if side not in ("src", "trg", "both"):
            raise Exception(f"'side' must be in ('src', 'trg', 'both'): '{side}'")

        # Get URLs which should be processed
        tokenized_url_and_dic = []
        orig_src_url = copy.copy(tokenized_src_url)
        orig_trg_url = copy.copy(tokenized_trg_url)
        min_hits = 1
        hits = 0

        if side == "src":
            tokenized_url_and_dic = [(tokenized_src_url, double_linked_freqs_src)]
        elif side == "trg":
            tokenized_url_and_dic = [(tokenized_trg_url, double_linked_freqs_trg)]
        elif side == "both":
            min_hits = 2
            tokenized_url_and_dic = [(tokenized_src_url, double_linked_freqs_src),
                                     (tokenized_trg_url, double_linked_freqs_trg)]
        else:
            raise Exception(f"Unknown side strategy: {side}")

        # Replace tokens with equal frequency
        for tokenized_url, double_linked_freqs in tokenized_url_and_dic:
            # sentence will modify either tokenized_src_url, tokenized_trg_url or both (i.e. is a reference)

            count = 0
            hit = False

            # Loop until any of the chosen words have an alternative, at most 'max_tries' times
            while count < max_tries and not hit:
                # Random number of words that will be replaced (at least 1)
                idx_words_to_replace = get_position_idxs_random(tokenized_url, percentage=percentage_tokens_affected,
                                                                min_elements=min_replacements)

                if len(idx_words_to_replace) >= min_replacements:
                    total_replacements = 0

                    for idx, wordpos in enumerate(idx_words_to_replace, 1):
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
                            alternatives = double_linked_freqs.get_words_for_occs(woccs, exact=False, fixed_limit=10 if woccs > 100 else None)

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

                        if len(idx_words_to_replace) - idx < min_replacements - total_replacements:
                            # We are sure we won't meet the requirements, so we can stop earlier
                            break

                    if total_replacements < min_replacements:
                        tokenized_src_url = orig_src_url
                        tokenized_trg_url = orig_trg_url

                count += 1

                if tokenized_src_url != orig_src_url or tokenized_trg_url != orig_trg_url:
                    # We got different URLs! 1 hit, at least (we might have had more since 'len(tokenized_url_and_dic) > 1' is possible)
                    # We can't restore original URLs because we might remove the modifications done by
                    #  other loop instance from 'tokenized_url_and_dic' element
                    hit = True

            if hit:
                hits += 1

        if hits > min_hits:
            logging.warning("hits > min_hits: %d > %d: it is not expected (side strategy: %s)", hits, min_hits, side)

        necessary_hits = hits >= min_hits

        return tokenized_src_url, tokenized_trg_url, necessary_hits

    side_priority = ("trg", "src", "both") # Priority for modifying src, trg or both URLs -> the priority is important since it will depend on 'limit_max_alignments_per_url'

    def process(src_url, trg_url, level, seed=None):
        utils.set_up_logging(level=level)

        if seed is not None:
            random.seed(seed)

        results = []

        for _ in range(limit_max_alignments_per_url):
            _results = []

            if side in ("all", "all-any"):
                hits = 0

                # Iterate through all the "side" strategies
                for idx in range(len(side_priority)):
                    _side = side_priority[idx % len(side_priority)] # Take a side taking into account the specified priority
                    _src_url, _trg_url, _, hit = apply_function_to_negative_sample_tokenized_urls(
                        src_url, trg_url,
                        lambda s, t: run(s, t, side=_side, min_replacements=min_replacements))

                    if hit:
                        _results.append((_src_url, _trg_url))

                        hits += 1
                    else:
                        if side == "all":
                            _results = []

                            break # The "all" strategy needs all hits, not any
            else:
                _src_url, _trg_url, _, hit = apply_function_to_negative_sample_tokenized_urls(
                    src_url, trg_url,
                    lambda s, t: run(s, t, side=side, min_replacements=min_replacements))

                if hit:
                    _results.append((_src_url, _trg_url))

            results.extend(_results)

        if len(results) == 0:
            logging.warning("Couldn't find words with the same frequency: couldn't generate negative sample for the provided URLs with the side '%s': ('%s', '%s')",
                            side, str(_src_url), str(_trg_url))

        return results

    _results = \
        joblib.Parallel(n_jobs=n_jobs, max_nbytes=None)( \
        joblib.delayed(process)(src_url, trg_url, logging.root.level, seed=random.randint(~sys.maxsize, sys.maxsize)) \
            for src_url, trg_url in parallel_urls)

    for _r in _results:
        for _src_url, _trg_url in _r:
            urls.add((_src_url, _trg_url))

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
        joblib.Parallel(n_jobs=n_jobs, max_nbytes=None)( \
        joblib.delayed(apply_function_to_negative_sample_tokenized_urls)(src_url, trg_url, run, seed=random.randint(~sys.maxsize, sys.maxsize),
                       level=logging.root.level) \
            for src_url, trg_url in parallel_urls for _ in range(limit_max_alignments_per_url))

    for _src_url, _trg_url, hit in _results:
        if hit:
            urls.add((_src_url, _trg_url))
        else:
            logging.warning("Couldn't generate negative samples removing random tokens: ('%s', '%s')", _src_url, _trg_url)

    common_last_checks(urls, parallel_urls)

    return list(urls)

_bow_logging_parallelization_variable_once = False
def get_negative_samples_intersection_metric(parallel_urls, limit_max_alignments_per_url=10, append_metric=False, n_jobs=1,
                                             apply_resource_forward=True):
    # Download NLTK model if not available
    utils.check_nltk_model("tokenizers/punkt", "punkt", download=True) # Download before parallel: https://github.com/nltk/nltk/issues/1576

    urls = set()
    max_pairs_to_be_generated = min(limit_max_alignments_per_url, len(parallel_urls) - 1) * len(parallel_urls)

    def tokenize_urls(src_url, trg_url, level):
        utils.set_up_logging(level=level)

        resource_idx_src_url = utils.get_idx_resource(src_url) if apply_resource_forward else 0
        resource_idx_trg_url = utils.get_idx_resource(trg_url) if apply_resource_forward else 0
        tokenized_src_url = tokenize(src_url[resource_idx_src_url:])
        tokenized_trg_url = tokenize(trg_url[resource_idx_trg_url:])

        return tokenized_src_url, tokenized_trg_url

    tokenized_urls = \
        joblib.Parallel(n_jobs=n_jobs, max_nbytes=None)( \
        joblib.delayed(tokenize_urls)(src_url, trg_url, logging.root.level) for src_url, trg_url in parallel_urls)

    def get_metrics(src_url, trg_url, idx_pair_src_url, idx_pair_trg_url, level=None):
        if level is not None:
            utils.set_up_logging(level=level)

        tokenized_src_url = set(tokenized_urls[idx_pair_src_url][0])
        tokenized_trg_url = set(tokenized_urls[idx_pair_trg_url][1])
        src_url_set = set(src_url)
        trg_url_set = set(trg_url)

        # Apply Jaccard (https://stats.stackexchange.com/a/290740)
        metric1_denominator = len(set.union(tokenized_src_url, tokenized_trg_url))
        metric2_denominator = len(set.union(src_url_set, trg_url_set))
        metric1 = (len(set.intersection(tokenized_src_url, tokenized_trg_url)) / metric1_denominator) if metric1_denominator != 0 else 0.0
        metric2 = (len(set.intersection(src_url_set, trg_url_set)) / metric2_denominator) if metric2_denominator != 0 else 0.0

        return trg_url, (metric1, metric2)

    global _bow_logging_parallelization_variable_once
    metrics_parallel = False # Parallelization disabled by default since it seems to be slower due to sorted()

    try:
        metrics_parallel = bool(int(os.environ["PUC_NSG_BOW_METRIC_PARALLEL"]))
    except ValueError:
        if not _bow_logging_parallelization_variable_once:
            logging.error("Envvar PUC_NSG_BOW_METRIC_PARALLEL was defined but couldn't be casted to int")
    except KeyError:
        pass

    if not _bow_logging_parallelization_variable_once:
        logging.debug("BOW metrics are going to be calculated using parallelization (envvar PUC_NSG_BOW_METRIC_PARALLEL): %s", metrics_parallel)

    _bow_logging_parallelization_variable_once = True
    duplicated_pairs = 0

    for idx_pair_src_url, parallel_urls_pair in enumerate(parallel_urls):
        src_url, _ = parallel_urls_pair

        if metrics_parallel:
            _results = \
                joblib.Parallel(n_jobs=n_jobs, max_nbytes=None)( \
                joblib.delayed(get_metrics)(src_url, trg_url, idx_pair_src_url, idx_pair_trg_url, level=logging.root.level) \
                    for idx_pair_trg_url, (_, trg_url) in enumerate(parallel_urls) if idx_pair_src_url != idx_pair_trg_url)
            sorted_trg_parallel_urls_dict = sorted(_results, key=lambda item: (item[1][0], item[1][1]), reverse=True)
        else:
            best_values = []

            for idx_pair_trg_url, (_, trg_url) in enumerate(parallel_urls):
                if idx_pair_src_url != idx_pair_trg_url:
                    metrics = get_metrics(src_url, trg_url, idx_pair_src_url, idx_pair_trg_url)
                    _trg_url, (metric1, metric2) = metrics
                    hit = False

                    for idx, (_, (_metric1, _metric2)) in enumerate(best_values):
                        if metric1 > _metric1:
                            hit = True

                            break
                        elif metric1 == _metric1 and metric2 > _metric2:
                            hit = True

                            break

                    if len(best_values) == 0:
                        best_values.append(metrics)
                    elif hit or len(best_values) < limit_max_alignments_per_url:
                        best_values.insert(idx, metrics)

                        if len(best_values) > limit_max_alignments_per_url:
                            best_values.pop()

            sorted_trg_parallel_urls_dict = best_values

        for idx, (trg_url, metrics) in enumerate(sorted_trg_parallel_urls_dict):
            if idx >= limit_max_alignments_per_url:
                break

            len_urls = len(urls)

            if append_metric:
                urls.add((src_url, trg_url, *metrics))
            else:
                urls.add((src_url, trg_url))

            duplicated_pairs += 1 if len(urls) == len_urls else 0

    logging.debug("Generated but duplicated (not added) pairs: %d", duplicated_pairs)

    if max_pairs_to_be_generated != len(urls) + duplicated_pairs:
        logging.error("The number of generated pairs is not the expected: %d was expected, but %d (unique) + %d (duplicated) were generated: bug?",
                      max_pairs_to_be_generated, len(urls), duplicated_pairs)

    common_last_checks(urls, parallel_urls)

    return list(urls)

def get_negative_samples_random(parallel_urls, limit_max_alignments_per_url=10):
    idxs = range(len(parallel_urls)) # Do not convert to list for performance reasons!
    urls = set()
    k = min(limit_max_alignments_per_url, len(parallel_urls))

    if len(parallel_urls) < limit_max_alignments_per_url:
        if len(parallel_urls) > 0:
            src_domain = extract(parallel_urls[0][0])[1]
            trg_domain = extract(parallel_urls[0][1])[1]

            logging.warning("Will not be possible to generate the required %d random pairs of URLs: "
                            "%d pairs will not be generated for the src/trg domain '%s'/'%s'",
                            limit_max_alignments_per_url, limit_max_alignments_per_url - len(parallel_urls),
                            src_domain, trg_domain)
        else:
            logging.warning("Will not be possible to generate the required %d random pairs of URLs: "
                            "%d pairs will not be generated since no parallel URLs were provided",
                            limit_max_alignments_per_url, limit_max_alignments_per_url - len(parallel_urls))

    for idx1 in idxs:
        max_alignments_per_url = limit_max_alignments_per_url
        sample_idxs = random.sample(idxs, k)

        for sort_idx2, idx2 in enumerate(sample_idxs, 1):
            if idx1 == idx2:
                # Skip parallel URLs
                max_alignments_per_url += 1

                continue

            if sort_idx2 > max_alignments_per_url:
                break

            src_url = parallel_urls[idx1][0]
            trg_url = parallel_urls[idx2][1]

            urls.add((src_url, trg_url))

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

def apply_function_to_negative_sample_tokenized_urls(src_url, trg_url, func, apply_resource_forward=True, seed=None, level=None):
    """
    Input is str URLs
    Input of 'func' is src and trg tokenized URLs
    Output of 'func' must be src and trg tuple of lists or strings
    Output is str URLs
    """
    if level is not None:
        utils.set_up_logging(level=level)

    if seed is not None:
        random.seed(seed)

    # Get tokens
    resource_idx_src_url = utils.get_idx_resource(src_url) if apply_resource_forward else 0
    resource_idx_trg_url = utils.get_idx_resource(trg_url) if apply_resource_forward else 0
    tokenized_src_url = tokenize(src_url[resource_idx_src_url:])
    tokenized_trg_url = tokenize(trg_url[resource_idx_trg_url:])

    # We want to apply the function to URL resource forward
    src_url_before_resource = src_url[:resource_idx_src_url]
    trg_url_before_resource = trg_url[:resource_idx_trg_url]

    # Apply function which will modify the tokens
    func_output = func(tokenized_src_url, tokenized_trg_url)
    tokenized_src_url, tokenized_trg_url = func_output[0], func_output[1]

    # Reconstruct
    src_url_again = src_url_before_resource + (''.join(tokenized_src_url) if isinstance(tokenized_src_url, list) else tokenized_src_url)
    trg_url_again = trg_url_before_resource + (''.join(tokenized_trg_url) if isinstance(tokenized_trg_url, list) else tokenized_trg_url)

    if src_url == src_url_again and trg_url == trg_url_again:
        # Do not add a negative sample which actually is a positive sample
        return src_url, trg_url, False, *func_output[2:]

    # Replace multiple '/' with one '/' (only the protocol has to have 2 '/')
    protocol_idx_src_url = utils.get_idx_after_protocol(src_url_again)
    protocol_idx_trg_url = utils.get_idx_after_protocol(trg_url_again)
    src_url_again = src_url_again[:protocol_idx_src_url] + re.sub(r'/+', r'/', src_url_again[protocol_idx_src_url:])
    trg_url_again = trg_url_again[:protocol_idx_trg_url] + re.sub(r'/+', r'/', trg_url_again[protocol_idx_trg_url:])

    return src_url_again, trg_url_again, True, *func_output[2:]

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
