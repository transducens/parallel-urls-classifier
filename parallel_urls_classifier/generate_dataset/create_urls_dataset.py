
import os
import sys
import random
import logging

import negative_samples_generator as nsg

logging.basicConfig(level=logging.INFO)

input_file_parallel_urls = sys.argv[1]
output_file_urls_prefix = sys.argv[2]

if "PYTHONHASHSEED" not in os.environ:
    logging.warning("You did not provide PYTHONHASHSEED: the results will not be deterministic")

random_seed = 71213

random.seed(random_seed)

train_perc = 0.8
dev_perc = 0.1
test_perc = 0.1
parallel_urls = {}
non_parallel_urls = {}
skipped_urls = 0
no_parallel_urls = 0
no_non_parallel_urls = 0

with open(input_file_parallel_urls) as f:
    for idx, url_pair in enumerate(f):
        url_pair = url_pair.strip().split('\t')

        assert len(url_pair) == 2, f"The provided line does not have 2 tab-separated values (line #{idx + 1})"

        if len(url_pair[0]) == 0 or len(url_pair[1]) == 0:
            logging.warning("Skipping line #%d because there are empty values", idx + 1)
            skipped_urls += 1

            continue
        if len(url_pair[0]) > 1000 or len(url_pair[1]) > 1000:
            logging.warning("Skipping line #%d because there are URLs too long (%d and %d)", idx + 1, len(url_pair[0]), len(url_pair[1]))
            skipped_urls += 1

            continue

        domains = ((url_pair[0] + '/').split('/')[2].replace('\t', ' '), (url_pair[1] + '/').split('/')[2].replace('\t', ' '))

        """
        if domains[0] != domains[1]:
            logging.debug("Skipping line #%d because the URLs do not belong to the same domain (%s vs %s)", idx + 1, domains[0], domains[1])
            skipped_urls += 1

            continue
        """

        domain = f"{domains[0]}\t{domains[1]}"

        if domain not in parallel_urls:
            parallel_urls[domain] = set()

        parallel_urls[domain].add((url_pair[0], url_pair[1]))

        no_parallel_urls += 1

    logging.info("Skipped lines: %d out of %d (%.2f %%)", skipped_urls, idx + 1, skipped_urls * 100.0 / (idx + 1))

logging.info("Loaded URLs (positive samples): %d", no_parallel_urls)
logging.info("Total domains (positive samples): %d", len(parallel_urls.keys()))

train_domains, train_max_idx = set(), int(train_perc * len(parallel_urls.keys()))
dev_domains, dev_max_idx = set(), train_max_idx + int(dev_perc * len(parallel_urls.keys()))
test_domains, test_max_idx = set(), len(parallel_urls.keys())
idx = 0

for idx, domain in enumerate(parallel_urls.keys()):
    if idx < train_max_idx:
        train_domains.add(domain)
    elif idx < dev_max_idx:
        dev_domains.add(domain)
    elif idx < test_max_idx:
        test_domains.add(domain)

logging.info("Train domains: %d", len(train_domains))
logging.info("Dev domains: %d", len(dev_domains))
logging.info("Test domains: %d", len(test_domains))

assert len(train_domains) + len(dev_domains) + len(test_domains) == len(parallel_urls.keys()), "Not all the domains have been set to a set"

def store_negative_samples(parallel_urls, non_parallel_filename, target_domains, limit_alignments=True, limit_max_alignments_per_url=10,
                           logging_cte=2, negative_samples_generator=nsg.get_negative_samples_random):
    no_parallel_domains = len(target_domains)
    no_non_parallel_urls = 0
    no_non_parallel_domains = 0
    last_perc_shown = -1

    with open(non_parallel_filename, "w") as f:
        for idx, domain in enumerate(target_domains):
            parallel_urls_domain = list(parallel_urls[domain]) # WARNING: you will need to set PYTHONHASHSEED if you want deterministic results across different executions
            negative_samples = negative_samples_generator(parallel_urls_domain, limit_alignments=limit_alignments, limit_max_alignments_per_url=limit_max_alignments_per_url)

            for src_url, trg_url in negative_samples:
                f.write(f"{src_url}\t{trg_url}\n")

            finished_perc = (idx + 1) * 100.0 / no_parallel_domains

            # Show every 0.02%
            if int(finished_perc * 100.0) % logging_cte == 0 and int(finished_perc * 100.0) != last_perc_shown:
                logging.info("%.2f %% of negative samples generated (%d out of %d domains were already processed): %d negative samples loaded",
                            finished_perc, idx + 1, no_parallel_domains, no_non_parallel_urls)
                last_perc_shown = int(finished_perc * 100.0)

            no_non_parallel_domains += 1 if len(negative_samples) != 0 else 0
            no_non_parallel_urls += len(negative_samples)

    return no_non_parallel_urls, no_non_parallel_domains

def store_dataset(parallel_urls, target_domains, parallel_filename, non_parallel_filename, logging_cte=2):
    no_parallel_urls = 0
    no_parallel_domains = len(target_domains)
    last_perc_shown = -1

    # Store parallel URLs
    with open(parallel_filename, "w") as f:
        for idx, domain in enumerate(target_domains):
            for url1, url2 in parallel_urls[domain]:
                f.write(f"{url1}\t{url2}\n")

                no_parallel_urls += 1

            finished_perc = (idx + 1) * 100.0 / no_parallel_domains

            # Show every (logging_cte / 100) %
            if int(finished_perc * 100.0) % logging_cte == 0 and int(finished_perc * 100.0) != last_perc_shown:
                logging.info("%.2f %% of positive samples generated (%d out of %d domains were already processed): %d positive samples loaded",
                            finished_perc, idx + 1, no_parallel_domains, no_parallel_urls)
                last_perc_shown = int(finished_perc * 100.0)

    logging.info("Total URLs for '%s' (positive samples): %d", parallel_filename, no_parallel_urls)
    logging.info("Total domains for '%s' (positive samples): %d", parallel_filename, no_parallel_domains)

    # TODO change to option with argparse
    #negative_samples_generator = nsg.get_negative_samples_random
    negative_samples_generator = nsg.get_negative_samples_intersection_metric

    # Create negative samples -> same domain and get all combinations (store non-parallel URLs)
    no_non_parallel_urls, no_non_parallel_domains = store_negative_samples(parallel_urls, non_parallel_filename, target_domains, limit_alignments=True, limit_max_alignments_per_url=10,
                                                                           logging_cte=logging_cte, negative_samples_generator=negative_samples_generator)

    logging.info("Total URLs for '%s' (negative samples): %d", non_parallel_filename, no_non_parallel_urls)
    logging.info("Total domains for '%s' (negative samples): %d", non_parallel_filename, no_non_parallel_domains)

if len(train_domains) == 0 or len(dev_domains) == 0 or len(test_domains) == 0:
    logging.warning("Some set has been detected to contain 0 domains (train, dev, test: %d, %d, %d): merging all the domains")

    all_parallel_urls = set()
    all_domain = "all"

    for domain in train_domains.union(dev_domains).union(test_domains):
        all_parallel_urls.update(parallel_urls[domain])

    all_parallel_urls = list(all_parallel_urls)

    logging.info("Processing %d URL pairs at once", len(all_parallel_urls))

    train_max_idx = int(train_perc * len(all_parallel_urls))
    dev_max_idx = train_max_idx + int(dev_perc * len(all_parallel_urls))
    test_max_idx = len(all_parallel_urls)

    store_dataset({all_domain: all_parallel_urls[0:train_max_idx]}, [all_domain], f"{output_file_urls_prefix}.parallel.train", f"{output_file_urls_prefix}.non-parallel.train", logging_cte=50)
    store_dataset({all_domain: all_parallel_urls[train_max_idx:dev_max_idx]}, [all_domain], f"{output_file_urls_prefix}.parallel.dev", f"{output_file_urls_prefix}.non-parallel.dev", logging_cte=100)
    store_dataset({all_domain: all_parallel_urls[dev_max_idx:test_max_idx]}, [all_domain], f"{output_file_urls_prefix}.parallel.test", f"{output_file_urls_prefix}.non-parallel.test", logging_cte=100)
else:
    store_dataset(parallel_urls, train_domains, f"{output_file_urls_prefix}.parallel.train", f"{output_file_urls_prefix}.non-parallel.train", logging_cte=50)
    store_dataset(parallel_urls, dev_domains, f"{output_file_urls_prefix}.parallel.dev", f"{output_file_urls_prefix}.non-parallel.dev", logging_cte=100)
    store_dataset(parallel_urls, test_domains, f"{output_file_urls_prefix}.parallel.test", f"{output_file_urls_prefix}.non-parallel.test", logging_cte=100)
