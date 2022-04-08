
import logging
import argparse

from transformers import AutoTokenizer

import utils

def main(args):
    dataset = args.dataset
    pretrained_model = args.pretrained_model
    threshold = args.threshold

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, max_length_tokens=int(1e30))

    tokenizer.max_model_input_sizes[pretrained_model] = int(1e30) # Fake max length of the model to avoid warnings

    print(f"src_url\ttrg_url\tpre_tokenized_urls\tno_tokens")

    for pre_tokenized_urls_batch, initial_urls_batch in utils.tokenize_batch_from_fd(dataset, tokenizer, args.batch_size, f=lambda u: utils.preprocess_url(u, remove_protocol_and_authority=args.remove_authority), return_urls=True):
        tokens_batch = tokenizer(pre_tokenized_urls_batch)["input_ids"]

        assert len(tokens_batch) == len(pre_tokenized_urls_batch), f"len(tokens_batch) != len(pre_tokenized_urls_batch): {len(tokens_batch)} vs {len(pre_tokenized_urls_batch)}"
        assert len(tokens_batch) == len(initial_urls_batch), f"len(tokens_batch) != len(initial_urls_batch): {len(tokens_batch)} vs {len(initial_urls_batch)}"

        for initial_urls, pre_tokenized_urls, tokens in zip(initial_urls_batch, pre_tokenized_urls_batch, tokens_batch):
            len_tokens = len(tokens) + 2 # add 2 tokens since eos and sep are added at the beginning and ending of the sentence respectively

            if threshold < 0 or len_tokens < threshold:
                continue

            print(f"{initial_urls[0]}\t{initial_urls[1]}\t{pre_tokenized_urls}\t{len_tokens}")

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Get tokens information from a dataset of parallel URLs")

    parser.add_argument('dataset', type=argparse.FileType('rt'), help="Filename with URLs (TSV format)")

    parser.add_argument('--batch-size', type=int, default=16, help="Batch size")
    parser.add_argument('--pretrained-model', default="xlm-roberta-base", help="Pretrained model")
    parser.add_argument('--remove-authority', action="store_true", help="Remove protocol and authority from provided URLs")
    parser.add_argument('--threshold', type=int, default=-1, help="Remove protocol and authority from provided URLs")

    parser.add_argument('-v', '--verbose', action="store_true", help="Verbose logging mode")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = initialization()

    utils.set_up_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    logging.debug("Arguments processed: {}".format(str(args)))

    main(args)