
import logging
import argparse

import parallel_urls_classifier.utils.utils as utils
import parallel_urls_classifier as puc
import parallel_urls_classifier.inference as puc_inference

import torch
import numpy as np
from flask import (
    Flask,
    request,
    jsonify,
)

app = Flask("parallel-urls-classifier-flask-server")

global_conf = {
    "model": None,
    "tokenizer": None,
    "device": None,
    "batch_size": None,
    "max_length_tokens": None,
    "amp_context_manager": None,
    "remove_authority": None,
    "remove_positional_data_from_resource": None,
    "parallel_likelihood": None,
    "url_separator": None,
}
logger = logging

@app.route('/hello-world', methods=['GET'])
def hello_world():
    return jsonify({"ok": "hello world! server is working!", "err": "null"})

@app.route('/inference', methods=['POST'])
def inference():
    if request.method == 'POST':
        return jsonify({"ok": "null", "err": "method: POST"})

    # Get parameters
    src_urls = request.form["src_urls"]
    trg_urls = request.form["trg_urls"]

    if not src_urls or not trg_urls:
        return jsonify({"ok": "null", "err": "'src_url' and 'trg_url' are mandatory fields and can't be empty"})

    if not isinstance(src_urls, list) or not isinstance(trg_urls, list):
        logger.warning("Single src and/or trg URL was provided instead of a batch: this will slow the inference")

        if not isinstance(src_urls, list):
            src_urls = [src_urls]
        if not isinstance(trg_urls, list):
            trg_urls = [trg_urls]

        if len(src_urls) != len(trg_urls):
            raise Exception(f"Different src and trg length: {len(src_urls)} vs {len(trg_urls)}")

    model = global_conf["model"]
    tokenizer = global_conf["tokenizer"]
    device = global_conf["device"]
    batch_size = global_conf["batch_size"]
    max_length_tokens = global_conf["max_length_tokens"]
    amp_context_manager = global_conf["amp_context_manager"]
    remove_authority = global_conf["remove_authority"]
    remove_positional_data_from_resource = global_conf["remove_positional_data_from_resource"]
    parallel_likelihood = global_conf["parallel_likelihood"]
    url_separator = global_conf["url_separator"]

    # Inference
    results = \
    puc_inference.non_interactive_inference(model, tokenizer, batch_size, max_length_tokens, device, amp_context_manager, src_urls, trg_urls,
                                            remove_authority=remove_authority, remove_positional_data_from_resource=remove_positional_data_from_resource,
                                            parallel_likelihood=parallel_likelihood, url_separator=url_separator)

    # Return results

    if len(results) != len(src_urls):
        return jsonify({"ok": "null", "err": f"results length mismatch with the provided URLs: {len(results)} vs {len(src_urls)}"})

    return jsonify({"ok": results, "err": "null"})

def main(args):
    model_input = args.model_input
    use_cuda = torch.cuda.is_available()
    force_cpu = args.force_cpu
    device = torch.device("cuda:0" if use_cuda and not force_cpu else "cpu")
    pretrained_model = args.pretrained_model

    global_conf["model"] = puc.load_model(model_input=model_input, device=device)
    global_conf["tokenizer"] = puc.load_tokenizer(pretrained_model)
    global_conf["device"] = device
    global_conf["batch_size"] = args.batch_size
    global_conf["max_length_tokens"] = args.max_length_tokens
    global_conf["amp_context_manager"] = puc.get_amp_context_manager(args.cuda_amp, force_cpu)
    global_conf["remove_authority"] = args.remove_authority
    global_conf["remove_positional_data_from_resource"] = not args.do_not_remove_positional_data_from_resource
    global_conf["parallel_likelihood"] = args.parallel_likelihood
    global_conf["url_separator"] = args.url_separator

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Parallel URLs classifier: flask server")

    parser.add_argument('model_input', help="Model input path which will be loaded")

    # TODO remove unused args
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size")
    parser.add_argument('--pretrained-model', default="xlm-roberta-base", help="Pretrained model")
    parser.add_argument('--max-length-tokens', type=int, default=256, help="Max. length for the generated tokens")
    parser.add_argument('--parallel-likelihood', action="store_true", help="Print parallel likelihood instead of classification string (inference)")
    parser.add_argument('--threshold', type=float, default=-np.inf, help="Only print URLs which have a parallel likelihood greater than the provided threshold (inference)")
    parser.add_argument('--remove-authority', action="store_true", help="Remove protocol and authority from provided URLs")
    parser.add_argument('--do-not-remove-positional-data-from-resource', action="store_true", help="Remove content after '#' in the resorce (e.g. https://www.example.com/resource#position -> https://www.example.com/resource)")
    parser.add_argument('--force-cpu', action="store_true", help="Run on CPU (i.e. do not check if GPU is possible)")
    parser.add_argument('--url-separator', default=' ', help="Separator to use when URLs are stringified")
    parser.add_argument('--cuda-amp', action="store_true", help="Use CUDA AMP (Automatic Mixed Precision)")

    parser.add_argument('-v', '--verbose', action="store_true", help="Verbose logging mode")

    args = parser.parse_args()

    print(args)

    return args

def cli():
    global logger

    args = initialization()
    logger = utils.set_up_logging_logger(logging.getLogger("parallel_urls_classifier.flask_server"), level=logging.DEBUG if args.verbose else logging.INFO)

    logger.debug("Arguments processed: {}".format(str(args)))

    puc.logger = logger

    main(args)

if __name__ == "__main__":
    cli()
