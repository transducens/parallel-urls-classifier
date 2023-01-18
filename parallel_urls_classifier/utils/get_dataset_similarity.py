
import os
import sys
import logging

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, f"{cdir}/../..")

from parallel_urls_classifier.tokenizer import tokenize

logging.basicConfig(level=logging.INFO)

final_intersection1, final_union1 = 0, 0
final_intersection2, final_union2 = 0, 0

# header
print("src_url\ttrg_url\tstringify_src_url\tstringify_trg_url\tintersection\tunion\tratio_intersection_union")

for idx, url_pair in enumerate(sys.stdin):
    url_pair = url_pair.strip().split('\t')

    assert len(url_pair) == 2, f"The provided line does not have 2 tab-separated values (line #{idx + 1})"

    src_url, trg_url = url_pair[0], url_pair[1]
    tokenized_src_url = tokenize(src_url)
    tokenized_trg_url = tokenize(trg_url)
    tokenized_src_url_set = set(tokenized_src_url)
    tokenized_trg_url_set = set(tokenized_trg_url)
    tokenized_src_url = ' '.join(tokenized_src_url)
    tokenized_trg_url = ' '.join(tokenized_trg_url)
    src_url_set = set(src_url)
    trg_url_set = set(trg_url)

    # Apply Jaccard (https://stats.stackexchange.com/a/290740)
    metric1_intersection = len(set.intersection(tokenized_src_url_set, tokenized_trg_url_set))
    metric2_intersection = len(set.intersection(src_url_set, trg_url_set))
    metric1_union = len(set.union(tokenized_src_url_set, tokenized_trg_url_set))
    metric2_union = len(set.union(src_url_set, trg_url_set))
    metric1 = (metric1_intersection / metric1_union) if metric1_union != 0 else 0.0
    metric2 = (metric2_intersection / metric2_union) if metric2_union != 0 else 0.0

    print(f"{src_url}\t{trg_url}\t{tokenized_src_url}\t{tokenized_trg_url}\t{metric1_intersection}\t{metric1_union}\t{metric1}\t"
          f"{metric2_intersection}\t{metric2_union}\t{metric2}")

    final_intersection += intersection
    final_union += union

logging.info("Total pairs: %d", idx + 1)

final_ratio1 = final_intersection1 / final_union1
final_ratio2 = final_intersection2 / final_union2

logging.info("Final intersection of tokens and chars: %d %d", final_intersection1, final_intersection2)
logging.info("Final union of tokens and chars: %d %d", final_union1, final_union2)
logging.info("Final ratio of tokens and chars (Jaccard): %.2f %.2f", final_ratio1, final_ratio2)
