
import os
import sys
import logging

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, f"{cdir}/../..")

from parallel_urls_classifier.tokenizer import tokenize

logging.basicConfig(level=logging.INFO)

final_intersection, final_union, final_ratio = 0, 0, 0.0

# header
print("src_url\ttrg_url\tstringify_src_url\tstringify_trg_url\tintersection\tunion\tratio_intersection_union")

for idx, url_pair in enumerate(sys.stdin):
    url_pair = url_pair.strip().split('\t')

    assert len(url_pair) == 2, f"The provided line does not have 2 tab-separated values (line #{idx + 1})"

    src_url, trg_url = url_pair[0], url_pair[1]
    tokenized_src_url = ' '.join(tokenize(src_url))
    tokenized_trg_url = ' '.join(tokenize(trg_url))
    src_set = set(tokenized_src_url.split(' '))
    trg_set = set(tokenized_trg_url.split(' '))
    intersection = len(src_set.intersection(trg_set))
    union = len(src_set.union(trg_set))
    ratio = intersection / union

    print(f"{src_url}\t{trg_url}\t{tokenized_src_url}\t{tokenized_trg_url}\t{intersection}\t{union}\t{ratio}")

    final_intersection += intersection
    final_union += union

logging.info("Total pairs: %d", idx + 1)

final_ratio = final_intersection / final_union

logging.info("Final intersection: %d", final_intersection)
logging.info("Final union: %d", final_union)
logging.info("Final ratio: %.2f", final_ratio)
