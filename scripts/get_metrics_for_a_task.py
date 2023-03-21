
import sys

import numpy as np

# It assumes monotonic data

sys.stdin.reconfigure(encoding='utf-8', errors="backslashreplace")

def mcc(tp, tn, fp, fn):
    almost_dividend = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return 100 * (tp * tn - fp * fn) / np.sqrt(almost_dividend) if almost_dividend != 0 else 0.0

def pos_and_neg_prec_recall_and_f1(tp, tn, fp, fn):
    results = {
        "pos_prec": 100 * tp / (tp + fp) if (tp + fp) != 0 else 100.0,
        "pos_recall": 100 * tp / (tp + fn) if (tp + fn) != 0 else 100.0,
        "neg_prec": 100 * tn / (tn + fn) if (tn + fn) != 0 else 100.0,
        "neg_recall": 100 * tn / (tn + fp) if (tn + fp) != 0 else 100.0,
    }

    results["pos_f1"] = 2 * results["pos_prec"] * results["pos_recall"] / (results["pos_prec"] + results["pos_recall"])
    results["neg_f1"] = 2 * results["neg_prec"] * results["neg_recall"] / (results["neg_prec"] + results["neg_recall"])

    return results

inference_file = sys.argv[1]
task_prefix = sys.argv[2]
gs_file = sys.argv[3]

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
        raise Exception(f"Different len(gs_pairs) and len(inference_pairs) for task {k}: {len(gs_pairs)} != {len(inference_pairs[k])}")

tp, tn, fp, fn = 0, 0, 0, 0
threshold = 0.5 # TODO parametrize?

for inference_pair, gs_pair in zip(inference_pairs[task_prefix], gs_pairs):
    inference_pair = inference_pair[1:]
    inference_pair[0] = float(inference_pair[0])

    if task_prefix == "urls":
        tp += 1 if gs_pair[2] == '1' and inference_pair[0] >= threshold else 0
        fp += 1 if gs_pair[2] == '0' and inference_pair[0] >= threshold else 0
        tn += 1 if gs_pair[2] == '0' and inference_pair[0] < threshold else 0
        fn += 1 if gs_pair[2] == '1' and inference_pair[0] < threshold else 0
    elif task_prefix == "language":
        tp += 1 if gs_pair[4] == gs_pair[6] and inference_pair[0] >= threshold else 0
        fp += 1 if gs_pair[4] != gs_pair[6] and inference_pair[0] >= threshold else 0
        tn += 1 if gs_pair[4] != gs_pair[6] and inference_pair[0] < threshold else 0
        fn += 1 if gs_pair[4] == gs_pair[6] and inference_pair[0] < threshold else 0
    elif task_prefix == "langid":
        tp += 1 if (gs_pair[4] == gs_pair[6] and gs_pair[2] == '1') and inference_pair[0] >= threshold else 0
        fp += 1 if (gs_pair[4] != gs_pair[6] or gs_pair[2] == '0') and inference_pair[0] >= threshold else 0
        tn += 1 if (gs_pair[4] != gs_pair[6] or gs_pair[2] == '0') and inference_pair[0] < threshold else 0
        fn += 1 if (gs_pair[4] == gs_pair[6] and gs_pair[2] == '1') and inference_pair[0] < threshold else 0
    else:
        raise Exception(f"Unknown task: {task_prefix}")

mcc_value = mcc(tp, tn, fp, fn)
pos_and_neg_prec_recall_and_f1_value = pos_and_neg_prec_recall_and_f1(tp, tn, fp, fn)
macro_f1 = (pos_and_neg_prec_recall_and_f1_value["pos_f1"] + pos_and_neg_prec_recall_and_f1_value["neg_f1"]) / 2.0

print(f"Conf. mat.: TP, TN, FP, FN: {tp}, {tn}, {fp}, {fn}")
print(f"Neg prec, recall and F1: {pos_and_neg_prec_recall_and_f1_value['neg_prec']} {pos_and_neg_prec_recall_and_f1_value['neg_recall']} {pos_and_neg_prec_recall_and_f1_value['neg_f1']}")
print(f"Pos prec, recall and F1: {pos_and_neg_prec_recall_and_f1_value['pos_prec']} {pos_and_neg_prec_recall_and_f1_value['pos_recall']} {pos_and_neg_prec_recall_and_f1_value['pos_f1']}")
print(f"Macro F1: {macro_f1}")
print(f"MCC: {mcc_value}")
