
import os
import sys
import logging
import argparse
import time

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, f"{cdir}/../..")

import parallel_urls_classifier.preprocess as preprocess
import parallel_urls_classifier.utils.utils as utils

from transformers import AutoTokenizer

def main(args):
    dataset = args.dataset
    pretrained_model = args.pretrained_model
    threshold = args.threshold
    lower = not args.do_not_lower

    for i in range(3):
        try:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model, max_length_tokens=int(1e30))

            break
        except Exception as e:
            if i == 3 - 1:
                logger.error("Couldn't load the tokenizer after 3 retries of 5 min")

                raise e

            time.sleep(60 * 5) # 5 min

    tokenizer.max_model_input_sizes[pretrained_model] = int(1e30) # Fake max length of the model to avoid warnings

    print(f"src_url\ttrg_url\tpre_tokenized_urls\tno_tokens")

    batch = utils.tokenize_batch_from_iterator(dataset, tokenizer, args.batch_size,
                                         f=lambda u: preprocess.preprocess_url(u, remove_protocol_and_authority=args.remove_authority,
                                                                               lower=lower),
                                         return_urls=True)

    for pre_tokenized_urls_batch, initial_urls_batch in batch:
        pre_tokenized_urls_batch = pre_tokenized_urls_batch["urls"]
        tokens_batch = tokenizer(pre_tokenized_urls_batch)["input_ids"]

        assert len(tokens_batch) == len(pre_tokenized_urls_batch), f"len(tokens_batch) != len(pre_tokenized_urls_batch): {len(tokens_batch)} vs {len(pre_tokenized_urls_batch)}"
        assert len(tokens_batch) == len(initial_urls_batch), f"len(tokens_batch) != len(initial_urls_batch): {len(tokens_batch)} vs {len(initial_urls_batch)}"

        for initial_urls, pre_tokenized_urls, tokens in zip(initial_urls_batch, pre_tokenized_urls_batch, tokens_batch):
            len_tokens = len(tokens)

            if threshold >= 0 and threshold < 0 or len_tokens < threshold:
                continue

            print(f"{initial_urls[0]}\t{initial_urls[1]}\t{pre_tokenized_urls}\t{len_tokens}")

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Get tokens information from a dataset of parallel URLs")

    parser.add_argument('dataset', type=argparse.FileType('rt', errors="backslashreplace"), help="Filename with URLs (TSV format)")

    parser.add_argument('--batch-size', type=int, default=16, help="Batch size")
    parser.add_argument('--pretrained-model', default="xlm-roberta-base", help="Pretrained model")
    parser.add_argument('--remove-authority', action="store_true", help="Remove protocol and authority from provided URLs")
    parser.add_argument('--threshold', type=int, default=-1, help="Minimum number of tokens in order to consider the pair of URLs")
    parser.add_argument('--do-not-lower', action="store_true", help="Do not lower URLs while preprocessing")

    parser.add_argument('-v', '--verbose', action="store_true", help="Verbose logging mode")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    # https://stackoverflow.com/questions/16549332/python-3-how-to-specify-stdin-encoding
    sys.stdin.reconfigure(encoding='utf-8', errors="backslashreplace")

    args = initialization()

    utils.set_up_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    logging.debug("Arguments processed: {}".format(str(args)))

    main(args)