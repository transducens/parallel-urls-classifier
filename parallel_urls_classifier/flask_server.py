
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

global_conf = {} # Empty since it will be filled once main is run
logger = logging.getLogger("parallel_urls_classifier")

# Disable (less verbose) 3rd party logging
logging.getLogger("werkzeug").setLevel(logging.WARNING)

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

    # Optional parameters
    try:
        if request.method == "GET":
            # GET method should be used only for testing purposes since HTML encoding is not being handled
            src_urls_lang = request.args.getlist("src_urls_lang")
            trg_urls_lang = request.args.getlist("trg_urls_lang")
        elif request.method == "POST":
            src_urls_lang = request.form.getlist("src_urls_lang")
            trg_urls_lang = request.form.getlist("trg_urls_lang")
        else:
            logger.warning("Unknown method: %s", request.method)

            return jsonify({"ok": "null", "err": f"unknown method: {request.method}"})

        if len(src_urls_lang) == 0:
            src_urls_lang = None
        if len(trg_urls_lang) == 0:
            trg_urls_lang = None
    except KeyError as e:
        src_urls_lang = None
        trg_urls_lang = None

    langs_provided = src_urls_lang is not None and trg_urls_lang is not None

    if not src_urls or not trg_urls:
        logger.warning("Empty src or trg urls: %s | %s", src_urls, trg_urls)

        return jsonify({"ok": "null", "err": "'src_urls' and 'trg_urls' are mandatory fields and can't be empty"})

    if not isinstance(src_urls, list) or not isinstance(trg_urls, list):
        logger.warning("Single src and/or trg URL was provided instead of a batch: this will slow the inference")

        if not isinstance(src_urls, list):
            src_urls = [src_urls]
        if not isinstance(trg_urls, list):
            trg_urls = [trg_urls]

    if len(src_urls) != len(trg_urls):
        return jsonify({"ok": "null", "err": f"different src and trg length: {len(src_urls)} vs {len(trg_urls)}"})

    if langs_provided:
        if not isinstance(src_urls_lang, list) or not isinstance(trg_urls_lang, list):
            if not isinstance(src_urls_lang, list):
                src_urls_lang = [src_urls_lang]
            if not isinstance(trg_urls_lang, list):
                trg_urls_lang = [trg_urls_lang]

        if len(src_urls_lang) != len(trg_urls_lang):
            return jsonify({"ok": "null", "err": f"different src and trg langs length: {len(src_urls_lang)} vs {len(trg_urls_lang)}"})
        if len(src_urls) != len(trg_urls_lang):
            return jsonify({"ok": "null", "err": f"different urls and langs length: {len(src_urls)} vs {len(trg_urls_lang)}"})

    logger.debug("Got (%d, %d) src and trg URLs", len(src_urls), len(trg_urls))

    base64_encoded = global_conf["expect_urls_base64"]

    if base64_encoded:
        try:
            src_urls = [base64.b64decode(f"{u.replace('_', '+')}==").decode("utf-8", errors="backslashreplace").replace('\n', ' ') for u in src_urls]
            trg_urls = [base64.b64decode(f"{u.replace('_', '+')}==").decode("utf-8", errors="backslashreplace").replace('\n', ' ') for u in trg_urls]
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
    if langs_provided:
        urls = [f"{src_url}\t{trg_url}\t{src_url_lang}\t{trg_url_lang}" for src_url, trg_url, src_url_lang, trg_url_lang in zip(src_urls, trg_urls, src_urls_lang, trg_urls_lang)]
    else:
        urls = [f"{src_url}\t{trg_url}" for src_url, trg_url in zip(src_urls, trg_urls)]

    results = get_results(urls)

    # Return results
    if len(results) != len(src_urls):
        logger.warning("Results length mismatch with the provided URLs (task '%s'): %d vs %d: %s vs %s",
                        task, len(results), len(src_urls), results, src_urls)

        return jsonify({
            "ok": "null",
            "err": f"results length mismatch with the provided URLs (task '{task}'): {len(results)} vs {len(src_urls)}",
        })

    results = [str(r) for r in results]

    logger.debug("Results: %s", results)

    return jsonify({
        "ok": results,
        "err": "null",
    })

def batch_prediction(urls):
    logger.debug("URLs batch size: %d", len(urls))

    src_urls, trg_urls = [], []
    src_urls_lang, trg_urls_lang = [], []
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
    auxiliary_tasks = global_conf["auxiliary_tasks"]
    target_task = global_conf["target_task"]

    for url in urls:
        fields = url.split('\t')

        src_urls.append(fields[0])
        trg_urls.append(fields[1])

        if len(fields) >= 4:
            src_urls_lang.append(fields[2])
            trg_urls_lang.append(fields[3])

    # Inference
    results = puc_inference.non_interactive_inference(
        model, tokenizer, batch_size, max_length_tokens, device, amp_context_manager, src_urls, trg_urls,
        remove_authority=remove_authority, remove_positional_data_from_resource=remove_positional_data_from_resource,
        parallel_likelihood=parallel_likelihood, url_separator=url_separator, lower=lower,
        auxiliary_tasks=auxiliary_tasks, src_urls_lang=src_urls_lang, trg_urls_lang=trg_urls_lang,
    )

    return results[target_task] # TODO do we need a list if the streamer is used (it seems so)?
                                # https://github.com/ShannonAI/service-streamer/issues/97

def main(args):
    model_input = args.model_input
    force_cpu = args.force_cpu
    use_cuda = utils.use_cuda(force_cpu=force_cpu)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    pretrained_model = args.pretrained_model
    flask_port = args.flask_port
    lower = args.lowercase
    auxiliary_tasks = args.auxiliary_tasks
    target_task = args.target_task
    regression = args.regression
    streamer_max_latency = args.streamer_max_latency
    run_flask_server = not args.do_not_run_flask_server
    disable_streamer = args.disable_streamer

    if not disable_streamer:
        logger.warning("Since streamer is enabled, you might get slightly different results: not recommended for production")
        # Related to https://discuss.pytorch.org/t/slightly-different-results-in-same-machine-and-gpu-but-different-order/173581

    if auxiliary_tasks is None:
        auxiliary_tasks = []

    logger.debug("Device: %s", device)

    if "model" not in global_conf:
        # model = load_model(all_tasks, all_tasks_kwargs, model_input=model_input, pretrained_model=pretrained_model, device=device)
        all_tasks = ["urls_classification"] + auxiliary_tasks
        all_tasks_kwargs = puc.load_tasks_kwargs(all_tasks, auxiliary_tasks, regression)
        global_conf["model"] = puc.load_model(all_tasks, all_tasks_kwargs, model_input=model_input,
                                              pretrained_model=pretrained_model, device=device)
    else:
        # We apply this step in order to avoid loading the model multiple times due to flask debug mode
        pass

    global_conf["tokenizer"] = puc.load_tokenizer(pretrained_model)
    global_conf["device"] = device
    global_conf["batch_size"] = args.batch_size
    global_conf["max_length_tokens"] = args.max_length_tokens
    global_conf["amp_context_manager"], _, _ = puc.get_amp_context_manager(args.cuda_amp, use_cuda)
    global_conf["remove_authority"] = args.remove_authority
    global_conf["remove_positional_data_from_resource"] = args.remove_positional_data_from_resource
    global_conf["parallel_likelihood"] = args.parallel_likelihood
    global_conf["url_separator"] = args.url_separator
    global_conf["streamer"] = ThreadedStreamer(batch_prediction, batch_size=args.batch_size, max_latency=streamer_max_latency)
    global_conf["disable_streamer"] = disable_streamer
    global_conf["expect_urls_base64"] = args.expect_urls_base64
    global_conf["lower"] = lower
    global_conf["auxiliary_tasks"] = auxiliary_tasks
    global_conf["target_task"] = target_task

    # Some guidance
    logger.info("Example: curl http://127.0.0.1:%d/hello-world", flask_port)
    logger.debug("Example: curl http://127.0.0.1:%d/inference -X POST -d \"src_urls=https://domain/resource1&trg_urls=https://domain/resource2\"", flask_port)

    if run_flask_server:
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
    parser.add_argument('--remove-positional-data-from-resource', action="store_true", help="Remove content after '#' in the resorce (e.g. https://www.example.com/resource#position -> https://www.example.com/resource)")
    parser.add_argument('--force-cpu', action="store_true", help="Run on CPU (i.e. do not check if GPU is possible)")
    parser.add_argument('--url-separator', default='/', help="Separator to use when URLs are stringified")
    parser.add_argument('--cuda-amp', action="store_true", help="Use CUDA AMP (Automatic Mixed Precision)")
    parser.add_argument('--disable-streamer', action="store_true", help="Do not use streamer (it might lead to slower inference and OOM errors)")
    parser.add_argument('--expect-urls-base64', action="store_true", help="Decode BASE64 URLs")
    parser.add_argument('--flask-port', type=int, default=5000, help="Flask port")
    parser.add_argument('--lowercase', action="store_true", help="Lowercase URLs while preprocessing")
    parser.add_argument('--auxiliary-tasks', type=str, nargs='*', choices=["mlm", "language-identification", "langid-and-urls_classification"],
                        help="Tasks which will try to help to the main task (multitasking)")
    parser.add_argument('--target-task', type=str, default="urls_classification",
                        help="Task which will be used as primary task and whose results will be used")
    parser.add_argument('--regression', action="store_true", help="Apply regression instead of binary classification")
    parser.add_argument('--streamer-max-latency', type=float, default=0.1,
                        help="Streamer max latency. You will need to modify this parameter if you want to increase the GPU usage")
    parser.add_argument('--do-not-run-flask-server', action="store_true", help="Do not run app.run")

    parser.add_argument('-v', '--verbose', action="store_true", help="Verbose logging mode")
    parser.add_argument('--flask-debug', action="store_true", help="Flask debug mode. Warning: this option might load the model multiple times")

    args = parser.parse_args()

    return args

def cli():
    global logger

    args = initialization()
    logger = utils.set_up_logging_logger(logging.getLogger("parallel_urls_classifier.flask_server"), level=logging.DEBUG if args.verbose else logging.INFO)

    logger.debug("Arguments processed: {}".format(str(args)))

    main(args)

    if not args.do_not_run_flask_server:
        logger.info("Bye!")
    else:
        logger.info("Execution has finished")

if __name__ == "__main__":
    cli()
