
import math
import base64
import logging
import argparse

import utils

def main(args):
    input_files = args.input
    src_url_idx = args.src_url_idx
    trg_url_idx = args.trg_url_idx
    domain_idx = args.domain_idx
    limit_non_parallel = args.limit_non_parallel

    urls = {}
    domains = {}

    for fd in input_files:
        path = fd.name
        filename = path.strip('/')[-1]

        logging.info("Processing '%s'", path)

        urls[filename] = {}

        for line in fd:
            line = line.rstrip('\n').split('\t')
            domain = line[domain_idx]
            src_url = line[src_url_idx]
            trg_url = line[trg_url_idx]

            try:
                urls[filename][domain]["src"].append(src_url)
                urls[filename][domain]["trg"].append(trg_url)
            except KeyError:
                urls[filename][domain] = {"src": [src_url], "trg": [trg_url]}

            try:
                domains[domain]
            except KeyError:
                domains[domain] = set()

            domains[domain].add(filename)

    domains_keys = sorted(domains.keys())
    labels = {"parallel": 0, "non-parallel": 0}
    non_parallel = 0
    max_non_parallel = 0

    for domain in domains_keys:
        filenames = sorted(domains[domain])

        logging.debug("Domain: %s (%d files)", domain, len(filename))

        for filename in filenames:
            max_non_parallel += len(urls[filename][domain]["src"])

            for src_idx, src_url in enumerate(urls[filename][domain]["src"]):
                for trg_idx, trg_url in enumerate(urls[filename][domain]["trg"]):
                    label = None

                    if src_idx == trg_idx:
                        label = "parallel"
                    else:
                        if abs(len(src_url) - len(trg_url)) >= 20:
                            label = "non-parallel"

                            if limit_non_parallel and non_parallel > max_non_parallel:
                                continue
                            else:
                                non_parallel += 1

                    if label:
                        labels[label] += 1
                        print(f"{label}\t{src_url}\t{trg_url}")

    logging.info("URLs (parallel, non-parallel): (%d, %d)", labels["parallel"], labels["non-parallel"])

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Get aligned URLs")

    parser.add_argument('--input', nargs='+', type=argparse.FileType('r'), required=True, help="TSV input files")
    parser.add_argument('--domain-idx', type=int, default=0, help="Source URL index")
    parser.add_argument('--src-url-idx', type=int, default=1, help="Source URL index")
    parser.add_argument('--trg-url-idx', type=int, default=2, help="Target URL index")

    parser.add_argument('--limit-non-parallel', action="store_true", help="Get the same quantity of non-parallel URLs that parallel URLs")

    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose logging mode")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = initialization()

    utils.set_up_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    logging.debug("Arguments processed: {}".format(str(args)))

    main(args)
