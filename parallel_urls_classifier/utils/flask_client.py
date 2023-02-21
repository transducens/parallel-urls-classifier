
import sys
import logging
import argparse
import base64
import json

import parallel_urls_classifier.utils.utils as utils

import requests
import joblib

logger = logging.getLogger("parallel_urls_classifier")

def fields2dict(fields, encode_base64=False):
    result = {
        "src_urls": [fields[0]],
        "trg_urls": [fields[1]],
    }

    if encode_base64:
        for k in ("src_urls", "trg_urls"):
            for idx, url in enumerate(result[k]):
                result[k][idx] = base64.b64encode(result[k][idx].encode("utf-8", errors="backslashreplace")).decode("utf-8", errors="backslashreplace")

    if len(fields) in (4, 5):
        result["src_langs"] = [fields[len(fields) - 2]]
        result["trg_langs"] = [fields[len(fields) - 1]]
    elif len(fields) in (6, 7):
        #result["src_langs"] = fields[len(fields) - 2]
        result["src_langs"] = [fields[len(fields) - 4]] # Src lang is always expected to be correct
        result["trg_langs"] = [fields[len(fields) - 1]]

    return result

def main(args):
    input_data = args.input_data
    urls_base64 = args.urls_base64
    flask_url = f"{args.flask_url.rstrip('/')}/inference"
    n_jobs = args.n_jobs

    if input_data == '-':
        input_data_fd = sys.stdin
    else:
        input_data_fd = open(input_data, 'r')

    data = []

    # Read data to be evaluated
    for line in input_data_fd:
        line = line.rstrip('\n').split('\t')

        if len(line) not in (2,3,4,5,6,7):
            raise Exception(f"Unexpected length: {len(line)}")

        if len(data) > 0 and len(line) != len(data[-1]):
            raise Exception(f"Different lengths: {len(line)} vs {len(data[-1])}")

        data.append(line)

    if input_data != '-':
        input_data_fd.close()

    # Get POST request parameters
    data = list(map(lambda d: fields2dict(d, encode_base64=urls_base64), data))

    # Get results from flask server
    def process_data(d):
        response = requests.post(flask_url, data=d)
        response = json.loads(response.text)

        if response["err"] != "null":
            logger.warning("Response error: %s", response["err"])
        else:
            if not isinstance(response["ok"], list):
                logger.error("A list of values were expected, but got: %s", response["ok"])
            else:
                if len(response["ok"]) != 1:
                    logger.error("Expected length was 1, but got %d", len(response["ok"]))
                else:
                    result = float(response["ok"][0])

                    return result

        return None

    start, end = 0, 0
    batch_size = 1000
    end = min(end + batch_size, len(data))

    while start != end:
        _results = joblib.Parallel(n_jobs=n_jobs, max_nbytes=None)(joblib.delayed(process_data)(d) for d in data[start:end])

        for _r in _results:
            print(_r)

        start = end
        end = min(end + batch_size, len(data))

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Parallel URLs classifier: flask server")

    parser.add_argument('input_data', help="Model input path which will be loaded")

    parser.add_argument('--urls-base64', action="store_true", help="Encode BASE64 URLs")
    parser.add_argument('--flask-url', type=str, default="http://127.0.0.1:5000", help="Flask server URL")
    parser.add_argument('--n-jobs', type=int, default=16, help="Number of jobs")

    parser.add_argument('-v', '--verbose', action="store_true", help="Verbose logging mode")

    args = parser.parse_args()

    return args

def cli():
    global logger

    args = initialization()
    logger = utils.set_up_logging_logger(logging.getLogger("parallel_urls_classifier.flask_client"), level=logging.DEBUG if args.verbose else logging.INFO)

    logger.debug("Arguments processed: {}".format(str(args)))

    main(args)

if __name__ == "__main__":
    sys.stdin.reconfigure(encoding='utf-8', errors="backslashreplace")

    cli()
