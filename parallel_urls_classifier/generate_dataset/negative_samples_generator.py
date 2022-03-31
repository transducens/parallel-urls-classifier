
import os
import sys
import random
import logging
import itertools

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, f"{cdir}/..")

import utils.utils as utils

def get_negative_samples_intersection_metric(parallel_urls, limit_alignments=True, limit_max_alignments_per_url=10,
                                             append_metric=False, log_debug=False):
    parallel_urls_stringify = {}
    urls = []

    # TODO something is WRONG! This generator is not returning the expected value, and even worse: is returning even parallel alignments!!!!
    # TODO why stringify_{src,trg}_url is not stringified?????

    for src_pair, trg_pair in itertools.combinations(parallel_urls, r=2):
        src_url = src_pair[0]
        trg_url = trg_pair[1]

        if src_url not in parallel_urls_stringify:
            parallel_urls_stringify[src_url] = {}

        stringify_src_url = utils.stringify_url(src_url)
        stringify_trg_url = utils.stringify_url(trg_url)
        metric = len(set(stringify_src_url.split(' ')).intersection(set(stringify_trg_url.split(' '))))

        parallel_urls_stringify[src_url][trg_url] = metric

        if log_debug:
            logging.debug("stringify_src_url <tab> stringify_trg_url <tab> metric: %s\t%s\t%d", stringify_src_url, stringify_trg_url, metric)

    for src_url in parallel_urls_stringify:
        sorted_trg_parallel_urls_stringify = sorted(parallel_urls_stringify[src_url].items(), key=lambda item: item[1], reverse=True)

        for idx, (trg_url, metric) in enumerate(sorted_trg_parallel_urls_stringify):
            if limit_alignments and idx >= limit_max_alignments_per_url:
                break

            if append_metric:
                urls.append((src_url, trg_url, metric))
            else:
                urls.append((src_url, trg_url))

    return urls

def get_negative_samples_random(parallel_urls, limit_alignments=True, limit_max_alignments_per_url=10):
    idxs2 = list(range(len(parallel_urls)))
    urls = []

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

            urls.append((src_url, trg_url))

    return urls

def show_info_from_fd(fd, generator, generator_kwargs=None, sample_size=None):
    urls = []

    for idx, url_pair in enumerate(fd):
        if sample_size is not None and idx >= sample_size:
            break

        url_pair = url_pair.strip().split('\t')

        assert len(url_pair) == 2, f"The provided line does not have 2 tab-separated values (line #{idx + 1})"

        urls.append((url_pair[0], url_pair[1]))

    if generator_kwargs:
        result = generator(urls, **generator_kwargs)
    else:
        result = generator(urls)

    for r in result:
        print(str(r))
