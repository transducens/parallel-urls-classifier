
import sys

import numpy as np

# It assumes monotonic data

sys.stdin.reconfigure(encoding='utf-8', errors="backslashreplace")

def mcc(tp, tn, fp, fn):
    almost_dividend = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return 100 * (tp * tn - fp * fn) / np.sqrt(float(almost_dividend)) if almost_dividend != 0 else 0.0

def pos_and_neg_prec_recall_and_f1(tp, tn, fp, fn):
    results = {
        "pos_prec": 100 * tp / (tp + fp) if (tp + fp) != 0 else 100.0,
        "pos_recall": 100 * tp / (tp + fn) if (tp + fn) != 0 else 100.0,
        "neg_prec": 100 * tn / (tn + fn) if (tn + fn) != 0 else 100.0,
        "neg_recall": 100 * tn / (tn + fp) if (tn + fp) != 0 else 100.0,
    }

    pos_dividend = results["pos_prec"] + results["pos_recall"]
    neg_dividend = results["neg_prec"] + results["neg_recall"]
    results["pos_f1"] = (2 * results["pos_prec"] * results["pos_recall"] / pos_dividend) if not np.isclose(pos_dividend, 0.0) else np.inf
    results["neg_f1"] = (2 * results["neg_prec"] * results["neg_recall"] / neg_dividend) if not np.isclose(neg_dividend, 0.0) else np.inf

    return results

inference_file = sys.argv[1]
task_prefix = sys.argv[2]
gs_file = sys.argv[3]
print_samples = sys.argv[4].split(',') if len(sys.argv) > 4 else [] # e.g. "tp", "tp,tn"

inference_pairs = {
    "urls": [],
    "language": [],
    "langid": [],
}
gs_pairs = []

if task_prefix not in inference_pairs.keys():
    raise Exception(f"Unexpected task. Allowed: {inference_pairs.keys()}")

with open(inference_file, 'r', encoding='utf-8', errors="backslashreplace") as fd:
    for l in fd:
        pair = l.rstrip('\n').split('\t')

        for k in inference_pairs.keys():
            if pair[0].startswith(k):
                inference_pairs[k].append(pair)

with open(gs_file, 'r', encoding='utf-8', errors="backslashreplace") as fd:
    for l in fd:
        pair = l.rstrip('\n').split('\t')

        gs_pairs.append(pair)

for k in inference_pairs.keys():
    if len(gs_pairs) != len(inference_pairs[k]):
        msg = f"Different len(gs_pairs) and len(inference_pairs) for task {k}: {len(gs_pairs)} != {len(inference_pairs[k])}"

        if k == task_prefix:
            raise Exception(msg)
        else:
            sys.stderr.write(f"WARNING: {msg}\n")

tp, tn, fp, fn = 0, 0, 0, 0
threshold = 0.5 # TODO parametrize?

if len(print_samples) != 0:
    print("task_prefix\tscore\tsrc_url\ttrg_url\tparallelness\ttrg_url_actual_lang\ttrg_url_lang_inference\tresult")

for inference_pair, gs_pair in zip(inference_pairs[task_prefix], gs_pairs):
    inference_pair = inference_pair[1:]
    inference_pair[0] = float(inference_pair[0])
    result = "none"

    if task_prefix == "urls":
        if gs_pair[2] == '1' and inference_pair[0] >= threshold:
            tp += 1
            result = "tp"
        if gs_pair[2] == '0' and inference_pair[0] >= threshold:
            fp += 1
            result = "fp"
        if gs_pair[2] == '0' and inference_pair[0] < threshold:
            tn += 1
            result = "tn"
        if gs_pair[2] == '1' and inference_pair[0] < threshold:
            fn += 1
            result = "fn"
    elif task_prefix == "language":
        if gs_pair[4] == gs_pair[6] and inference_pair[0] >= threshold:
            tp += 1
            result = "tp"
        if gs_pair[4] != gs_pair[6] and inference_pair[0] >= threshold:
            fp += 1
            result = "fp"
        if gs_pair[4] != gs_pair[6] and inference_pair[0] < threshold:
            tn += 1
            result = "tn"
        if gs_pair[4] == gs_pair[6] and inference_pair[0] < threshold:
            fn += 1
            result = "fn"
    elif task_prefix == "langid":
        if (gs_pair[4] == gs_pair[6] and gs_pair[2] == '1') and inference_pair[0] >= threshold:
            tp += 1
            result = "tp"
        if (gs_pair[4] != gs_pair[6] or gs_pair[2] == '0') and inference_pair[0] >= threshold:
            fp += 1
            result = "fp"
        if (gs_pair[4] != gs_pair[6] or gs_pair[2] == '0') and inference_pair[0] < threshold:
            tn += 1
            result = "tn"
        if (gs_pair[4] == gs_pair[6] and gs_pair[2] == '1') and inference_pair[0] < threshold:
            fn += 1
            result = "fn"
    else:
        raise Exception(f"Unknown task: {task_prefix}")

    if len(print_samples) != 0 and (result in print_samples or '-' in print_samples):
        inference_pair = '\t'.join(map(lambda s: str(s), inference_pair))

        print(f"{task_prefix}\t{inference_pair}\t{gs_pair[2]}\t{gs_pair[4]}\t{gs_pair[6]}\t{result}")

mcc_value = mcc(tp, tn, fp, fn)
pos_and_neg_prec_recall_and_f1_value = pos_and_neg_prec_recall_and_f1(tp, tn, fp, fn)
macro_f1 = (pos_and_neg_prec_recall_and_f1_value["pos_f1"] + pos_and_neg_prec_recall_and_f1_value["neg_f1"]) / 2.0

sys.stderr.write(f"Conf. mat.: TP, TN, FP, FN: {tp} {tn} {fp} {fn}\n")
sys.stderr.write(f"Neg prec, recall and F1: {round(pos_and_neg_prec_recall_and_f1_value['neg_prec'], 2)} {round(pos_and_neg_prec_recall_and_f1_value['neg_recall'], 2)} {round(pos_and_neg_prec_recall_and_f1_value['neg_f1'], 2)}\n")
sys.stderr.write(f"Pos prec, recall and F1: {round(pos_and_neg_prec_recall_and_f1_value['pos_prec'], 2)} {round(pos_and_neg_prec_recall_and_f1_value['pos_recall'], 2)} {round(pos_and_neg_prec_recall_and_f1_value['pos_f1'], 2)}\n")
sys.stderr.write(f"Macro F1: {round(macro_f1, 2)}\n")
sys.stderr.write(f"MCC: {round(mcc_value, 2)}\n")
