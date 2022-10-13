
import os
import sys
import random
import logging
import argparse

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, f"{cdir}/../..")

import parallel_urls_classifier.generate_dataset.negative_samples_generator as nsg
import parallel_urls_classifier.utils.utils as utils

import numpy as np

def store_negative_samples(parallel_urls, non_parallel_filename, target_domains, unary_generator, logging_cte=2):
    no_parallel_domains = len(target_domains)
    no_non_parallel_urls = 0
    no_non_parallel_domains = 0
    last_perc_shown = -1

    with open(non_parallel_filename, "a") as f:
        for idx, domain in enumerate(target_domains):
            parallel_urls_domain = list(parallel_urls[domain]) # WARNING: you will need to set PYTHONHASHSEED if you want deterministic results across different executions
            negative_samples = unary_generator(parallel_urls_domain)

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

def get_unary_generator(generator, limit_alignments=True, limit_max_alignments_per_url=10, extra_kwargs={}):
    return lambda data: generator(data, limit_alignments=limit_alignments,
                                  limit_max_alignments_per_url=limit_max_alignments_per_url, **extra_kwargs)


def store_dataset(parallel_urls, target_domains, parallel_filename, non_parallel_filename, logging_cte=2,
                  negative_samples_generator=["random"], max_negative_samples_alignments=10):
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

    with open(non_parallel_filename, "w") as f:
        # Create file and remove content if exists
        pass

    # Generate negative samples
    for idx, generator in enumerate(negative_samples_generator, 1):
        if generator != "none":
            extra_kwargs = {}

            if generator == "random":
                negative_samples_generator_f = nsg.get_negative_samples_random
            elif generator == "bow-overlapping-metric":
                negative_samples_generator_f = nsg.get_negative_samples_intersection_metric
            elif generator == "remove-random-tokens":
                negative_samples_generator_f = nsg.get_negative_samples_remove_random_tokens
            else:
                logging.warning("Generator %d: unknown negative samples generator (%s): skipping", idx, generator)

                continue

            unary_generator = get_unary_generator(negative_samples_generator_f, limit_alignments=True,
                                                  limit_max_alignments_per_url=max_negative_samples_alignments,
                                                  extra_kwargs=extra_kwargs)

            # Create negative samples -> same domain and get all combinations (store non-parallel URLs)
            no_non_parallel_urls, no_non_parallel_domains = \
                store_negative_samples(parallel_urls, non_parallel_filename, target_domains, unary_generator, logging_cte=logging_cte)

            logging.info("Generator %d: total URLs for '%s' (negative samples): %d", idx, non_parallel_filename, no_non_parallel_urls)
            logging.info("Generator %d: total domains for '%s' (negative samples): %d", idx, non_parallel_filename, no_non_parallel_domains)
        else:
            logging.debug("Generator %d: skip negative samples generation", idx)

def main(args):
    input_file_parallel_urls = args.input_file_parallel_urls
    output_file_urls_prefix = args.output_files_prefix
    negative_samples_generator = args.generator_technique
    generate_negative_samples = not args.do_not_generate_negative_samples
    seed = args.seed
    max_negative_samples_alignments = args.max_negative_samples_alignments
    same_authority = args.same_authority
    train_perc, dev_perc, test_perc = args.sets_percentage

    if not isinstance(negative_samples_generator, list):
        negative_samples_generator = list(negative_samples_generator)

    if not np.isclose(sum(args.sets_percentage), 1.0):
        raise Exception("The provided sets percentages do not sum up to 1.0")

    if "PYTHONHASHSEED" not in os.environ:
        logging.warning("You did not provide PYTHONHASHSEED: the results will not be deterministic")

    if seed >= 0:
        random.seed(seed)

    if not generate_negative_samples:
        negative_samples_generator = ["none"] # Force "none" generator in order to do not create negative samples

    parallel_urls = {}
    skipped_urls = 0
    no_parallel_urls = 0

    for idx, url_pair in enumerate(input_file_parallel_urls):
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

        if same_authority and domains[0] != domains[1]:
            logging.debug("Skipping line #%d because the URLs do not belong to the same domain (%s vs %s)", idx + 1, domains[0], domains[1])
            skipped_urls += 1

            continue

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
    all_domains = list(parallel_urls.keys())

    random.shuffle(all_domains) # Shuffle domains in order to avoid dependency between the data and the order it was provided

    for idx, domain in enumerate(all_domains):
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

    common_kwargs = {"negative_samples_generator": negative_samples_generator,
                     "max_negative_samples_alignments": max_negative_samples_alignments}

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

        store_dataset({all_domain: all_parallel_urls[0:train_max_idx]}, [all_domain], f"{output_file_urls_prefix}.parallel.train", f"{output_file_urls_prefix}.non-parallel.train", logging_cte=50, **common_kwargs)
        store_dataset({all_domain: all_parallel_urls[train_max_idx:dev_max_idx]}, [all_domain], f"{output_file_urls_prefix}.parallel.dev", f"{output_file_urls_prefix}.non-parallel.dev", logging_cte=100, **common_kwargs)
        store_dataset({all_domain: all_parallel_urls[dev_max_idx:test_max_idx]}, [all_domain], f"{output_file_urls_prefix}.parallel.test", f"{output_file_urls_prefix}.non-parallel.test", logging_cte=100, **common_kwargs)
    else:
        store_dataset(parallel_urls, train_domains, f"{output_file_urls_prefix}.parallel.train", f"{output_file_urls_prefix}.non-parallel.train", logging_cte=50, **common_kwargs)
        store_dataset(parallel_urls, dev_domains, f"{output_file_urls_prefix}.parallel.dev", f"{output_file_urls_prefix}.non-parallel.dev", logging_cte=100, **common_kwargs)
        store_dataset(parallel_urls, test_domains, f"{output_file_urls_prefix}.parallel.test", f"{output_file_urls_prefix}.non-parallel.test", logging_cte=100, **common_kwargs)

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Create URLs dataset from parallel samples")

    parser.add_argument('input_file_parallel_urls', type=argparse.FileType('rt'), help="Input TSV file with parallel URLs")
    parser.add_argument('output_files_prefix', help="Output files prefix")

    parser.add_argument('--generator-technique', choices=["none", "random", "bow-overlapping-metric", "remove-random-tokens"],
                        default="random", nargs='+', help="Strategy to create negative samples from positive samples")
    parser.add_argument('--max-negative-samples-alignments', type=int, default=3, help="Max. number of alignments of negative samples per positive samples per generator")
    parser.add_argument('--do-not-generate-negative-samples', action='store_true', help="Do not generate negative samples. Useful if you only want to split the data in train/dev/test")
    parser.add_argument('--same-authority', action='store_true', help="Skip pair of URLs with different authority")
    parser.add_argument('--sets-percentage', type=float, nargs=3, default=[0.8, 0.1, 0.1], help="Train, dev and test percentages")

    parser.add_argument('--force-non-deterministic', action='store_true', help="If PYTHONHASHSEED is not set, it will be set in order to obtain deterministic results. If this flag is set, this action will not be done")
    parser.add_argument('--seed', type=int, default=71213, help="Seed in order to have deterministic results (fully guaranteed if you also set PYTHONHASHSEED envvar). Set a negative number in order to disable this feature")
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose logging mode")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = initialization()

    utils.set_up_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    if "PYTHONHASHSEED" not in os.environ and not args.force_non_deterministic:
        # TODO it closes opened temporary named pipe files -> the execution crashes -> how to fix and still keep this behavior?
        # Wrapper call in order to define PYTHONHASHSEED (https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program)

        PYTHONHASHSEED_value = args.seed

        logging.warning("PYTHONHASHSEED not set: using seed: %d", args.seed)

        import subprocess

        subprocess.run([sys.executable] + sys.argv, env={**dict(os.environ), **{"PYTHONHASHSEED": str(PYTHONHASHSEED_value)}})
    else:
        logging.debug("Arguments processed: {}".format(str(args)))

        main(args)
