
import re
import os
import sys
import random
import logging
import itertools

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, f"{cdir}/../..")

from parallel_urls_classifier.tokenizer import tokenize

def common_last_checks(negative_samples_set, parallel_urls_set):
    urls_len = len(negative_samples_set)

    # Update reference set removing parallel pairs, if any
    negative_samples_set.difference_update(parallel_urls_set)

    urls_overlap = urls_len - len(negative_samples_set)

    if urls_overlap > 0:
        logging.warning("Bug? Parallel and non-parallel URLs sets overlap > 0: "
                        "this might happen if you have provided >1 pair of URLs where are >1 translation for the same document: "
                        "%d", urls_overlap)

def get_negative_samples_remove_random_tokens(parallel_urls, limit_alignments=True, limit_max_alignments_per_url=10, remove_percentage=0.4):
    if remove_percentage < 0.0 or remove_percentage > 1.0:
        raise Exception(f"0.0 <= remove_percentage <= 1.0: {remove_percentage}")

    parallel_urls_dict = {}
    urls = set()

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

    def run(src_url, trg_url):
        h = hash(f"{src_url}\t{trg_url}")

        # Get tokens
        resource_idx_src_url = get_idx_resource(src_url)
        resource_idx_trg_url = get_idx_resource(trg_url)
        tokenized_src_url = tokenize(src_url[resource_idx_src_url:])
        tokenized_trg_url = tokenize(trg_url[resource_idx_trg_url:])

        # We want to remove from URL resource forward
        src_url_before_resource = src_url[:resource_idx_src_url]
        trg_url_before_resource = trg_url[:resource_idx_trg_url]

        # Remove elements
        src_url_remove_idxs = reversed([idx for idx in range(len(tokenized_src_url)) if random.random() <= remove_percentage])
        trg_url_remove_idxs = reversed([idx for idx in range(len(tokenized_trg_url)) if random.random() <= remove_percentage])

        for idx in src_url_remove_idxs:
            tokenized_src_url.pop(idx)
        for idx in trg_url_remove_idxs:
            tokenized_trg_url.pop(idx)

        # Reconstruct
        src_url_again = src_url_before_resource + ''.join(tokenized_src_url)
        trg_url_again = trg_url_before_resource + ''.join(tokenized_trg_url)

        if hash(f"{src_url_again}\t{trg_url_again}") == h:
            # Do not add a negative sample which actually is a positive sample
            return src_url, trg_url

        # Replace multiple '/' with one '/'
        src_url_again = re.sub(r'/+', r'/', src_url_again)
        trg_url_again = re.sub(r'/+', r'/', trg_url_again)

        return src_url_again, trg_url_again

    return random_combinations(parallel_urls, limit_alignments=limit_alignments,
                               limit_max_alignments_per_url=limit_max_alignments_per_url,
                               binary_callback=run)

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
