
import sys
import random
import logging

logging.basicConfig(level=logging.INFO)

input_file_parallel_urls = sys.argv[1]
output_file_urls_prefix = sys.argv[2]

random_seed = 71213

random.seed(random_seed)

train_perc = 0.8
dev_perc = 0.1
test_perc = 0.1
urls = []

with open(input_file_parallel_urls) as f:
    for idx, url_pair in enumerate(f):
        url_pair = url_pair.strip().split('\t')

        assert len(url_pair) == 2, f"The provided line does not have 2 tab-separated values (line #{idx + 1})"

        urls.append(url_pair)

train_max_idx = int(train_perc * len(urls))
dev_max_idx = train_max_idx + int(dev_perc * len(urls))
test_max_idx = len(urls)

def store_dataset(parallel_urls, filename_prefix):
    parallel_filename = f"{filename_prefix}.parallel"
    non_parallel_filename = f"{filename_prefix}.non-parallel"
    no_parallel_urls = 0

    # Store parallel URLs
    with open(parallel_filename, "w") as f:
        for url1, url2 in parallel_urls:
            f.write(f"{url1}\t{url2}\n")

            no_parallel_urls += 1

    logging.info("Total URLs for '%s' (positive samples): %d", parallel_filename, no_parallel_urls)

    # Create negative samples -> same domain and get all combinations
    no_non_parallel_urls = 0
    limit_alignments = True

    # Store non-parallel URLs
    with open(non_parallel_filename, "w") as f:
        idxs2 = list(range(len(parallel_urls)))

        for idx1 in range(len(parallel_urls)):
            max_alignments_per_url = 10

            random.shuffle(idxs2)

            for sort_idx2, idx2 in enumerate(idxs2):
                if idx1 >= idx2:
                    # Skip parallel URLs and pairs which have been already seen before (get only combinations)
                    max_alignments_per_url += 1
                    continue

                if limit_alignments and sort_idx2 >= max_alignments_per_url:
                    # Try to avoid very large combinations
                    break

                pair1 = parallel_urls[idx1]
                pair2 = parallel_urls[idx2]
                url1 = pair1[0]
                url2 = pair2[1]

                f.write(f"{url1}\t{url2}\n")

                no_non_parallel_urls += 1

    logging.info("Total URLs for '%s' (negative samples): %d", non_parallel_filename, no_non_parallel_urls)

store_dataset(urls[0:train_max_idx], f"{output_file_urls_prefix}.train")
store_dataset(urls[train_max_idx:dev_max_idx], f"{output_file_urls_prefix}.dev")
store_dataset(urls[dev_max_idx:test_max_idx], f"{output_file_urls_prefix}.test")
