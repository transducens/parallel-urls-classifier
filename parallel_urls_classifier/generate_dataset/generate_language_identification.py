
import os
import sys
import random
import logging
import argparse

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, f"{cdir}/../..")

import parallel_urls_classifier.utils.utils as utils

def main(args):
    input_file_parallel_urls = args.input_file_parallel_urls
    output_files_prefix = args.output_files_prefix
    src_url_lang = args.src_url_lang
    trg_url_lang = args.trg_url_lang
    generator_technique = args.generator_technique
    max_negative_samples_alignments = args.max_negative_samples_alignments
    wrong_set_of_langs = [] if args.wrong_set_of_langs is None else args.wrong_set_of_langs
    seed = args.seed
    add_actual_langs = not args.do_not_add_actual_langs

    if seed >= 0:
        random.seed(seed)

    if "PYTHONHASHSEED" not in os.environ:
        logging.warning("You did not provide PYTHONHASHSEED: the results will not be deterministic")

    if src_url_lang == trg_url_lang and "same-lang-in-both-sides" in generator_technique:
            logging.error("Since src_url_lang == trg_url_lang, 'same-lang-in-both-sides' is going to be removed from --generator-technique"
                          " in order to avoid inserting false negatives")

            generator_technique.remove("same-lang-in-both-sides")

    pairs = set()

    logging.info("Generating positive samples...")

    # Read URLs and create positive samples
    for idx, pair_urls in enumerate(input_file_parallel_urls, 1):
        pair_urls = pair_urls.rstrip('\n').split('\t')

        if len(pair_urls) != 2:
            raise Exception(f"Pair #{idx} doesn't have 2 fields, but {len(pair_urls)}")

        pairs.add('\t'.join(pair_urls) + (f"\t{src_url_lang}\t{trg_url_lang}" * 2 if add_actual_langs else 1) + "\t1")

    negative_pairs = {}

    if max_negative_samples_alignments <= 0:
        generator_technique = [] # Force the non-creation of negative samples

        logging.warning("Negative samples will not be generated since --max-negative-samples-alignments <= 0")

    logging.info("Generating negative samples...")

    start_idx_pairs = 0
    end_idx_pairs = 4 if add_actual_langs else 2

    # Negative samples
    for generator in generator_technique:
        logging.info("Generating negative samples: %s", generator)

        if generator == "none":
            pass
        elif generator.startswith("random"):
            if generator in negative_pairs:
                logging.warning("Same generator for generating negative samples provided multiple times: %s", generator)

            negative_pairs[generator] = set()

            if generator == "random":
                random_func_generator = (lambda: '\t'.join(random.choices(wrong_set_of_langs, k=2)))
            elif generator == "random-src":
                random_func_generator = (lambda: f"{random.choice(wrong_set_of_langs)}\t{trg_url_lang}")
            elif generator == "random-trg":
                random_func_generator = (lambda: f"{src_url_lang}\t{random.choice(wrong_set_of_langs)}")
            else:
                raise Exception(f"Unknown 'random' generator: {generator}")

            if len(wrong_set_of_langs) == 0:
                logging.error("Negative samples generator 'random' needs wrong language ids, but none were provided")
            else:
                for pair in pairs:
                    _pair = pair.split('\t')[start_idx_pairs:end_idx_pairs]
                    random_pairs = set(
                        ['\t'.join(_pair) + '\t' + random_func_generator() + "\t0" \
                            for _ in range(max_negative_samples_alignments)])

                    for random_pair in random_pairs:
                        _random_pair = random_pair.split('\t')
                        _src_url = _random_pair[end_idx_pairs + 0]
                        _trg_url = _random_pair[end_idx_pairs + 1]

                        if _src_url == src_url_lang and _trg_url == trg_url_lang:
                            # We don't want to add an actually positive sample (i.e. false negative)
                            pass
                        else:
                            negative_pairs[generator].add(random_pair)
        elif generator == "swap-langs":
            # max_negative_samples_alignments don't apply here
            negative_pairs["swap-langs"] = set()

            for pair in pairs:
                negative_pairs["swap-langs"].add('\t'.join(pair.split('\t')[start_idx_pairs:end_idx_pairs]) + f"\t{trg_url_lang}\t{src_url_lang}\t0")
        elif generator == "same-lang-in-both-sides":
            negative_pairs["same-lang-in-both-sides"] = set()

            for pair in pairs:
                _pair = '\t'.join(pair.split('\t')[start_idx_pairs:end_idx_pairs])

                if max_negative_samples_alignments == 1:
                    if random.random() < 0.5:
                        negative_pairs["same-lang-in-both-sides"].add(_pair + f"\t{src_url_lang}\t{src_url_lang}\t0")
                    else:
                        negative_pairs["same-lang-in-both-sides"].add(_pair + f"\t{trg_url_lang}\t{trg_url_lang}\t0")
                else:
                    negative_pairs["same-lang-in-both-sides"].add(_pair + f"\t{src_url_lang}\t{src_url_lang}\t0")
                    negative_pairs["same-lang-in-both-sides"].add(_pair + f"\t{trg_url_lang}\t{trg_url_lang}\t0")
        else:
            raise Exception(f"Unknown generator: {generator}")

    logging.info("Writing positive and negative samples...")

    # Write positive samples
    with open(f"{output_files_prefix}.positive", 'w') as f:
        for pair in pairs:
            f.write(f"{pair}\n")

    # Write negative samples
    for generator, negative_samples in negative_pairs.items():
        with open(f"{output_files_prefix}.negative.generator_{generator}", 'w') as f:
            for negative_sample in negative_samples:
                f.write(f"{negative_sample}\n")

    logging.info("Done!")

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Generate language identification data for a set of URLs")
    parser.add_argument('input_file_parallel_urls', type=argparse.FileType('rt', errors="backslashreplace"), help="Input TSV file with parallel URLs")
    parser.add_argument('output_files_prefix', help="Output files prefix")
    parser.add_argument('src_url_lang', help="Language of the URL of the 1st column")
    parser.add_argument('trg_url_lang', help="Language of the URL of the 2nd column")

    parser.add_argument('--generator-technique', choices=["none", "random", "swap-langs", "same-lang-in-both-sides", "random-src", "random-trg"],
                        default=["random"], nargs='+',
                        help="Strategy to create negative samples from positive samples")
    parser.add_argument('--max-negative-samples-alignments', type=int, default=3, help="Max. number of alignments of negative samples per positive samples per generator")
    parser.add_argument('--wrong-set-of-langs', nargs='*', default=["en", "fr", "tr", "sl", "hr", "mk", "is", "mt", "bg"],
                        help="Set of language identifiers which are wrong for src and trg URLs. This will be useful for some operations (e.g. generate negative samples using 'random')")
    parser.add_argument('--do-not-add-actual-langs', action='store_true', help="Do not add additional columns with the actual languages")

    parser.add_argument('--seed', type=int, default=71213, help="Seed in order to have deterministic results (fully guaranteed if you also set PYTHONHASHSEED envvar). Set a negative number in order to disable this feature")
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose logging mode")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    # https://stackoverflow.com/questions/16549332/python-3-how-to-specify-stdin-encoding
    sys.stdin.reconfigure(encoding='utf-8', errors="backslashreplace")

    args = initialization()

    utils.set_up_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    logging.debug("Arguments processed: {}".format(str(args)))

    main(args)
