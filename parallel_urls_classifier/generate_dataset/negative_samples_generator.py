
import os
import sys
import random
import logging
import itertools

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, f"{cdir}/..")

#import utils.utils as utils

from nltk.tokenize import RegexpTokenizer

tokenize = RegexpTokenizer(r'[^\W_]+|[^\w\s]+|_').tokenize # Similar to wordpunct_tokenize but '_' aware

def get_negative_samples_intersection_metric(parallel_urls, limit_alignments=True, limit_max_alignments_per_url=10,
                                             append_metric=False):
    parallel_urls_stringify = {}
    urls = set()

    for src_pair, trg_pair in itertools.combinations(parallel_urls, r=2):
        src_url = src_pair[0]
        trg_url = trg_pair[1]

        if src_url not in parallel_urls_stringify:
            parallel_urls_stringify[src_url] = {}

        tokenized_src_url = tokenize(src_url)
        tokenized_trg_url = tokenize(trg_url)
        metric1 = len(set(tokenized_src_url).intersection(set(tokenized_trg_url)))
        metric2 = len(set(src_url).intersection(set(trg_url)))

        parallel_urls_stringify[src_url][trg_url] = (metric1, metric2)

    for src_url in parallel_urls_stringify:
        sorted_trg_parallel_urls_stringify = sorted(parallel_urls_stringify[src_url].items(), key=lambda item: (item[1][0], item[1][1]), reverse=True)

        for idx, (trg_url, metrics) in enumerate(sorted_trg_parallel_urls_stringify):
            if limit_alignments and idx >= limit_max_alignments_per_url:
                break

            if append_metric:
                urls.add((src_url, trg_url, *metrics))
            else:
                urls.add((src_url, trg_url))

    urls_len = len(urls)

    urls.difference_update(parallel_urls)

    urls_overlap = urls_len - len(urls)

    if urls_overlap > 0:
        logging.warning("Bug? Parallel and non-parallel URLs sets overlap > 0: %d", urls_overlap)

    return list(urls)

def get_negative_samples_random(parallel_urls, limit_alignments=True, limit_max_alignments_per_url=10):
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

            urls.add((src_url, trg_url))

    urls_len = len(urls)

    urls.difference_update(parallel_urls)

    urls_overlap = urls_len - len(urls)

    if urls_overlap > 0:
        logging.warning("Bug? Parallel and non-parallel URLs sets overlap > 0: %d", urls_overlap)

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
