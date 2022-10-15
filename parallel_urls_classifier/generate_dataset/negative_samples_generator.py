
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
from parallel_urls_classifier.word_freqs_double_linked import WordFreqDistDoubleLinked

def common_last_checks(negative_samples_set, parallel_urls_set):
    urls_len = len(negative_samples_set)

    # Update reference set removing parallel pairs, if any
    negative_samples_set.difference_update(parallel_urls_set)

    urls_overlap = urls_len - len(negative_samples_set)

    if urls_overlap > 0:
        logging.warning("Bug? Parallel and non-parallel URLs sets overlap > 0: "
                        "this might happen if you have provided >1 pair of URLs where are >1 translation for the same document: "
                        "%d", urls_overlap)

def get_negative_samples_replace_freq_words(parallel_urls, limit_alignments=True, limit_max_alignments_per_url=10,
                                            src_monolingual_file='', trg_monolingual_file=''):
    """
    Original function from https://github.com/bitextor/bicleaner-ai/blob/master/bicleaner_ai/training.py
    """
    if not utils.exists(src_monolingual_file) or not utils.exists(trg_monolingual_file):
        raise Exception(f"Either src, trg or both monolingual files does not exist: ('{src_monolingual_file}', '{trg_monolingual_file}')")

    double_linked_freqs_src = WordFreqDistDoubleLinked(src_monolingual_file)
    double_linked_freqs_trg = WordFreqDistDoubleLinked(trg_monolingual_file)
    urls = set()

    def run(tokenized_src_url, tokenized_trg_url, side='trg', max_tries=6, percentage_tokens_affected=0.5):
        if side not in ("src", "trg", "both"):
            raise Exception(f"'side' must be in ('src', 'trg', 'both'): '{side}'")

        # Get URLs which should be processed
        sentences = []
        orig_src_url = copy.copy(tokenized_src_url)
        orig_trg_url = copy.copy(tokenized_trg_url)

        if side == "src":
            sentences = [(tokenized_src_url, double_linked_freqs_src)]
        elif side == "trg":
            sentences = [(tokenized_trg_url, double_linked_freqs_trg)]
        elif side == "both":
            sentences = [(tokenized_src_url, double_linked_freqs_src),
                         (tokenized_trg_url, double_linked_freqs_trg)]

        # Replace tokens with equal frequency
        for sentence, double_linked_freqs in sentences:
            # sentence will modify either tokenized_src_url, tokenized_trg_url or both (i.e. is a reference)

            count = 0

            # Loop until any of the chosen words have an alternative, at most 'max_tries' times
            while True:
                # Random number of words that will be replaced (at least 1)
                idx_words_to_replace = get_position_idxs_random(sentence, percentage=percentage_tokens_affected, min_elements=1)

                for wordpos in idx_words_to_replace:
                    w = sentence[wordpos]
                    wfreq = double_linked_freqs.get_word_freq(w)
                    alternatives = double_linked_freqs.get_words_for_freq(wfreq)

                    if alternatives is not None:
                        alternatives = list(sorted(alternatives))

                        # Avoid replace with the same word
                        if w.lower() in alternatives:
                            alternatives.remove(w.lower())
                        if not alternatives == []:
                            sentence[wordpos] = random.choice(alternatives)

                            # Restore starting capital letter
                            # TODO better handling? If (w[1] == '.' and w[2].isupper()) or w[1].isupper() might not be necessary to capitalize but toupper() until (not character and not '.') is found for the 1st part or (not character) is found for the 2nd part of the if statement (i.e. initials)
                            if wordpos == 0 and w[0].isupper():
                                sentence[wordpos] = sentence[wordpos].capitalize()

                count += 1

                if tokenized_src_url != orig_src_url or tokenized_trg_url != orig_trg_url:
                    break
                elif count >= max_tries:
                    tokenized_src_url = orig_src_url
                    tokenized_trg_url = orig_trg_url

                    logging.warning("Couldn't find words with the same frequency: return original src and trg URLs: ('%s', '%s')",
                                    str(tokenized_src_url), str(tokenized_trg_url))

                    break

        return tokenized_src_url, tokenized_trg_url

    side_priority = ("trg", "src", "both") # Priority for modifying src, trg or both URLs -> the priority is important since it will depend on 'limit_max_alignments_per_url'

    for src_url, trg_url in parallel_urls:
        for idx in range(limit_max_alignments_per_url):
            side = side_priority[idx % len(side_priority)] # Take a side taking into account the specified priority
            src_url_src, trg_url_src = apply_function_to_negative_sample_tokenized_urls(src_url, trg_url, lambda s, t: run(s, t, side=side), "replace_freq_words")

            urls.add(src_url, trg_url)

    common_last_checks(urls, parallel_urls)

    return list(urls)

def get_negative_samples_remove_random_tokens(parallel_urls, limit_alignments=True, limit_max_alignments_per_url=10, remove_percentage=0.5):
    if remove_percentage < 0.0 or remove_percentage > 1.0:
        raise Exception(f"0.0 <= remove_percentage <= 1.0: {remove_percentage}")

    parallel_urls_dict = {}
    urls = set()

    def run(tokenized_src_url, tokenized_trg_url):
        # Remove elements (at least 1)
        src_url_remove_idxs = get_position_idxs_random(tokenized_src_url, percentage=remove_percentage, min_elements=1)
        trg_url_remove_idxs = get_position_idxs_random(tokenized_trg_url, percentage=remove_percentage, min_elements=1)

        for idx in src_url_remove_idxs:
            tokenized_src_url.pop(idx)
        for idx in trg_url_remove_idxs:
            tokenized_trg_url.pop(idx)

        return tokenized_src_url, tokenized_trg_url

    for src_url, trg_url in parallel_urls:
        for _ in range(limit_max_alignments_per_url):
            src_url, trg_url = apply_function_to_negative_sample_tokenized_urls(src_url, trg_url, run, "remove_random_tokens")

            urls.add(src_url, trg_url)

    common_last_checks(urls, parallel_urls)

    return list(urls)

def get_negative_samples_intersection_metric(parallel_urls, limit_alignments=True, limit_max_alignments_per_url=10,
                                             append_metric=False):
    parallel_urls_dict = {}
    urls = set()

    for src_pair, trg_pair in itertools.combinations(parallel_urls, r=2):
        src_url = src_pair[0]
        trg_url = trg_pair[1]

        if src_url not in parallel_urls_dict:
            parallel_urls_dict[src_url] = {}

        tokenized_src_url = tokenize(src_url)
        tokenized_trg_url = tokenize(trg_url)
        metric1 = len(set(tokenized_src_url).intersection(set(tokenized_trg_url)))
        metric2 = len(set(src_url).intersection(set(trg_url)))
        parallel_urls_dict[src_url][trg_url] = (metric1, metric2)

    for src_url in parallel_urls_dict:
        sorted_trg_parallel_urls_dict = sorted(parallel_urls_dict[src_url].items(), key=lambda item: (item[1][0], item[1][1]), reverse=True)

        for idx, (trg_url, metrics) in enumerate(sorted_trg_parallel_urls_dict):
            if limit_alignments and idx >= limit_max_alignments_per_url:
                break

            if append_metric:
                urls.add((src_url, trg_url, *metrics))
            else:
                urls.add((src_url, trg_url))

    common_last_checks(urls, parallel_urls)

    return list(urls)

def get_negative_samples_random(parallel_urls, limit_alignments=True, limit_max_alignments_per_url=10):
    return random_combinations(parallel_urls, limit_alignments=limit_alignments,
                               limit_max_alignments_per_url=limit_max_alignments_per_url)

def random_combinations(parallel_urls, limit_alignments=True, limit_max_alignments_per_url=10, binary_callback=None, **kwargs):
    idxs2 = list(range(len(parallel_urls)))
    urls = set()

    for idx1 in range(len(parallel_urls)):
        max_alignments_per_url = limit_max_alignments_per_url

        random.shuffle(idxs2)

        for sort_idx2, idx2 in enumerate(idxs2):
            if idx1 >= idx2:
                # Skip parallel URLs and pairs which have been already seen before (get only combinations)
                max_alignments_per_url += 1
                continue

            if limit_alignments and sort_idx2 >= max_alignments_per_url:
                # Try to avoid very large combinations
                break

            src_pair = parallel_urls[idx1]
            trg_pair = parallel_urls[idx2]
            src_url = src_pair[0]
            trg_url = trg_pair[1]

            if binary_callback:
                src_url, trg_url = binary_callback(src_url, trg_url, **kwargs)

            urls.add((src_url, trg_url))

    common_last_checks(urls, parallel_urls)

    return list(urls)

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

def get_idx_resource(url):
        idx = 0

        if url.startswith("http://") and not url.startswith("http:///"):
            idx += 7
        if url.startswith("https://") and not url.startswith("https:///"):
            idx += 8

        idx = url.find('/', idx)

        if idx == -1:
            return len(url) # There is no resource

        return idx + 1

def apply_function_to_negative_sample_tokenized_urls(src_url, trg_url, func, strategy_description, apply_resource_forward=True):
    """
    Input is str URLs
    Input of 'func' is src and trg tokenized URLs
    Output of 'func' must be src and trg tuple
    Output is str URLs
    """
    h = hash(f"{src_url}\t{trg_url}")

    # Get tokens
    resource_idx_src_url = get_idx_resource(src_url) if apply_resource_forward else 0
    resource_idx_trg_url = get_idx_resource(trg_url) if apply_resource_forward else 0
    tokenized_src_url = tokenize(src_url[resource_idx_src_url:])
    tokenized_trg_url = tokenize(trg_url[resource_idx_trg_url:])

    # We want to apply the function to URL resource forward
    src_url_before_resource = src_url[:resource_idx_src_url]
    trg_url_before_resource = trg_url[:resource_idx_trg_url]

    # Apply function which will modify the tokens
    tokenized_src_url, tokenized_trg_url = func(tokenized_src_url, tokenized_trg_url)

    # Reconstruct
    src_url_again = src_url_before_resource + ''.join(tokenized_src_url)
    trg_url_again = trg_url_before_resource + ''.join(tokenized_trg_url)

    if hash(f"{src_url_again}\t{trg_url_again}") == h:
        # Do not add a negative sample which actually is a positive sample
        logging.warning("Couldn't generate a negative sample using '%s': return original src and trg URLs: ('%s', '%s')",
                        strategy_description, src_url, trg_url)

        return src_url, trg_url

    # Replace multiple '/' with one '/'
    src_url_again = re.sub(r'/+', r'/', src_url_again)
    trg_url_again = re.sub(r'/+', r'/', trg_url_again)

    return src_url_again, trg_url_again

def get_position_idxs_random(l, percentage=0.5, min_elements=1, sort=True, reverse=True):
    idxs = []

    if percentage < 0.0:
        percentage = 0.0

        logging.warning("Percentage set to 0.0")

    # We want a distribution with the percentage specified, so we need to re-generate the indexes if needed
    # If we wanted not to generate a distribution with the specified percentage, there are more efficient ways to do it (e.g. iterate through the indexes and add if random < percentage)
    while len(idxs) < min_elements:
        idxs = reversed([idx for idx in range(len(l)) if random.random() <= percentage])

    if sort:
        idxs = sorted(idxs)

    if reverse:
        idxs = reversed(idxs)

    return idxs
