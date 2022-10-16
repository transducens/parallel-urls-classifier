
import logging
import argparse
import json
import base64

import parallel_urls_classifier.utils.utils as utils
import parallel_urls_classifier.parallel_urls_classifier as puc
import parallel_urls_classifier.inference as puc_inference

import torch
import numpy as np
from flask import (
    Flask,
    request,
    jsonify,
)
from service_streamer import ThreadedStreamer

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
    "streamer": None,
    "disable_streamer": None,
    "expect_urls_base64": None,
    "lower": None,
}
logger = logging

@app.route('/', methods=['GET'])
def info():
    available_routes = json.dumps(
        {
            "/hello-world": ["GET"],
            "/inference": ["GET", "POST"],
        },
        indent=4).replace('\n', '<br/>').replace(' ', '&nbsp;')

    return f"Available routes:<br/>{available_routes}"

@app.route('/hello-world', methods=["GET"])
def hello_world():
    return jsonify({"ok": "hello world! server is working!", "err": "null"})

@app.route('/inference', methods=["GET", "POST"])
def inference():
    if request.method not in ("GET", "POST"):
        return jsonify({"ok": "null", "err": "method is not: GET, POST"})

    # Get parameters
    try:
        if request.method == "GET":
            # GET method should be used only for testing purposes since HTML encoding is not being handled
            src_urls = request.args.getlist("src_urls")
            trg_urls = request.args.getlist("trg_urls")
        elif request.method == "POST":
            src_urls = request.form.getlist("src_urls")
            trg_urls = request.form.getlist("trg_urls")
        else:
            logger.warning("Unknown method: %s", request.method)

            return jsonify({"ok": "null", "err": f"unknown method: {request.method}"})
    except KeyError as e:
        logger.warning("KeyError: %s", e)

        return jsonify({"ok": "null", "err": f"could not get some mandatory field: 'src_urls' and 'trg_urls' are mandatory"})

    if not src_urls or not trg_urls:
        logger.warning("Empty src or trg urls")

        return jsonify({"ok": "null", "err": "'src_url' and 'trg_url' are mandatory fields and can't be empty"})

    if not isinstance(src_urls, list) or not isinstance(trg_urls, list):
        logger.warning("Single src and/or trg URL was provided instead of a batch: this will slow the inference")

        if not isinstance(src_urls, list):
            src_urls = [src_urls]
        if not isinstance(trg_urls, list):
            trg_urls = [trg_urls]

    if len(src_urls) != len(trg_urls):
        return jsonify({"ok": "null", "err": f"different src and trg length: {len(src_urls)} vs {len(trg_urls)}"})

    logger.debug("Got (%d, %d) src and trg URLs", len(src_urls), len(trg_urls))

    base64_encoded = global_conf["expect_urls_base64"]

    if base64_encoded:
        try:
            src_urls = [base64.b64decode(f"{u}==").decode("utf-8", errors="backslashreplace").replace('\n', ' ') for u in src_urls]
            trg_urls = [base64.b64decode(f"{u}==").decode("utf-8", errors="backslashreplace").replace('\n', ' ') for u in trg_urls]
        except Exception as e:
            logger.error("Exception when decoding BASE64: %s", e)

            return jsonify({"ok": "null", "err": "error decoding BASE64 URLs"})

    src_urls = [u.replace('\t', ' ') for u in src_urls]
    trg_urls = [u.replace('\t', ' ') for u in trg_urls]

    for src_url, trg_url in zip(src_urls, trg_urls):
        logger.debug("'src<tab>trg' URLs: %s\t%s", src_url, trg_url)

    # Inference

    disable_streamer = global_conf["disable_streamer"]
    get_results = global_conf["streamer"].predict if not disable_streamer else batch_prediction
    # We need one element per pair in order to do not break the streamer (it doesn't handle right the parallelism if there isn't 1:1 elements)
    urls = [f"{src_url}\t{trg_url}" for src_url, trg_url in zip(src_urls, trg_urls)]

    results = get_results(urls)

    # Return results

    if len(results) != len(src_urls):
        logger.warning("Results length mismatch with the provided URLs: %d vs %d: %s vs %s", len(results), len(src_urls), results, src_urls)

        return jsonify({"ok": "null", "err": f"results length mismatch with the provided URLs: {len(results)} vs {len(src_urls)}"})

    results = [str(r) for r in results]

    logger.debug("Results: %s", results)

    return jsonify({"ok": results, "err": "null"})

def batch_prediction(urls):
    logger.debug("URLs batch size: %d", len(urls))

    src_urls, trg_urls = [], []
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
    lower = global_conf["lower"]

    for url in urls:
        src_url, trg_url = url.split('\t')

        src_urls.append(src_url)
        trg_urls.append(trg_url)

    # Inference
    results = \
        puc_inference.non_interactive_inference(
            model, tokenizer, batch_size, max_length_tokens, device, amp_context_manager, src_urls, trg_urls,
            remove_authority=remove_authority, remove_positional_data_from_resource=remove_positional_data_from_resource,
            parallel_likelihood=parallel_likelihood, url_separator=url_separator, lower=lower)

    return results

def main(args):
    model_input = args.model_input
    use_cuda = torch.cuda.is_available()
    force_cpu = args.force_cpu
    device = torch.device("cuda:0" if use_cuda and not force_cpu else "cpu")
    pretrained_model = args.pretrained_model
    flask_port = args.flask_port
    lower = args.do_not_lower

    logger.debug("Device: %s", device)

    global_conf["model"] = puc.load_model(model_input=model_input, device=device) if not global_conf["model"] else global_conf["model"]
    global_conf["tokenizer"] = puc.load_tokenizer(pretrained_model)
    global_conf["device"] = device
    global_conf["batch_size"] = args.batch_size
    global_conf["max_length_tokens"] = args.max_length_tokens
    global_conf["amp_context_manager"] = puc.get_amp_context_manager(args.cuda_amp, force_cpu)
    global_conf["remove_authority"] = args.remove_authority
    global_conf["remove_positional_data_from_resource"] = not args.do_not_remove_positional_data_from_resource
    global_conf["parallel_likelihood"] = args.parallel_likelihood
    global_conf["url_separator"] = args.url_separator
    global_conf["streamer"] = ThreadedStreamer(batch_prediction, batch_size=args.batch_size)
    global_conf["disable_streamer"] = args.disable_streamer
    global_conf["expect_urls_base64"] = args.expect_urls_base64
    global_conf["lower"] = lower

    # Run flask server
    app.run(debug=args.flask_debug, port=flask_port)

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Parallel URLs classifier: flask server")

    parser.add_argument('model_input', help="Model input path which will be loaded")

    parser.add_argument('--batch-size', type=int, default=16, help="Batch size")
    parser.add_argument('--pretrained-model', default="xlm-roberta-base", help="Pretrained model")
    parser.add_argument('--max-length-tokens', type=int, default=256, help="Max. length for the generated tokens")
    parser.add_argument('--parallel-likelihood', action="store_true", help="Print parallel likelihood instead of classification string (inference)")
    parser.add_argument('--threshold', type=float, default=-np.inf, help="Only print URLs which have a parallel likelihood greater than the provided threshold (inference)")
    parser.add_argument('--remove-authority', action="store_true", help="Remove protocol and authority from provided URLs")
    parser.add_argument('--do-not-remove-positional-data-from-resource', action="store_true", help="Remove content after '#' in the resorce (e.g. https://www.example.com/resource#position -> https://www.example.com/resource)")
    parser.add_argument('--force-cpu', action="store_true", help="Run on CPU (i.e. do not check if GPU is possible)")
    parser.add_argument('--url-separator', default='/', help="Separator to use when URLs are stringified")
    parser.add_argument('--cuda-amp', action="store_true", help="Use CUDA AMP (Automatic Mixed Precision)")
    parser.add_argument('--disable-streamer', action="store_true", help="Do not use streamer (it might lead to slower inference and OOM errors)")
    parser.add_argument('--expect-urls-base64', action="store_true", help="Decode BASE64 URLs")
    parser.add_argument('--flask-port', type=int, default=5000, help="Flask port")
    parser.add_argument('--do-not-lower', action="store_true", help="Do not lower URLs while preprocessing")

    parser.add_argument('-v', '--verbose', action="store_true", help="Verbose logging mode")
    parser.add_argument('--flask-debug', action="store_true", help="Flask debug mode. Warning: this option might load the model multiple times")

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

    logger.info("Bye!")

if __name__ == "__main__":
    cli()
