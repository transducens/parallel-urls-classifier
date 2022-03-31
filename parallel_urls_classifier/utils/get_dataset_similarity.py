
import sys
import logging

import utils

logging.basicConfig(level=logging.INFO)

final_intersection, final_union, final_ratio = 0, 0, 0.0

# header
print("src_url\ttrg_url\tstringify_src_url\tstringify_trg_url\tintersection\tunion\tratio_intersection_union")

for idx, url_pair in enumerate(sys.stdin):
    url_pair = url_pair.strip().split('\t')

    assert len(url_pair) == 2, f"The provided line does not have 2 tab-separated values (line #{idx + 1})"

    src_url, trg_url = url_pair[0], url_pair[1]
    stringify_src_url = utils.stringify_url(src_url)
    stringify_trg_url = utils.stringify_url(trg_url)
    src_set = set(stringify_src_url.split(' '))
    trg_set = set(stringify_trg_url.split(' '))
    intersection = len(src_set.intersection(trg_set))
    union = len(src_set.union(trg_set))
    ratio = intersection / union

    print(f"{src_url}\t{trg_url}\t{stringify_src_url}\t{stringify_trg_url}\t{intersection}\t{union}\t{ratio}")

    final_intersection += intersection
    final_union += union

logging.info("Total pairs: %d", idx + 1)

final_ratio = final_intersection / final_union

logging.info("Final intersection: %d", final_intersection)
logging.info("Final union: %d", final_union)
logging.info("Final ratio: %.2f", final_ratio)
