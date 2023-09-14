
import sys

fn = sys.argv[1]
data = {}
websites = set()
all_processed_warcs = set()
warc2websites = {}

batch_size = 50

with open(fn) as fd:
  next(fd) # skip header

  for l in fd:
    l = l.rstrip("\r\n").split('\t')
    website = l[0]
    processed_warcs = int(l[1])
    actual_docs_cld2 = int(l[2])
    actual_docs_classifier = int(l[3])
    actual_docs_none = int(l[4])
    parallel_docs_cld2 = int(l[5])
    parallel_docs_classifier = int(l[6])
    parallel_docs_none = int(l[7])

#    if actual_docs_cld2 % batch_size != 0 or actual_docs_classifier % batch_size != 0 or actual_docs_none % batch_size != 0:
#      continue

    websites.add(website)
    all_processed_warcs.add(processed_warcs)

    if processed_warcs not in warc2websites:
      warc2websites[processed_warcs] = set()

    warc2websites[processed_warcs].add(website)

    if website not in data:
      data[website] = {}

    data[website][processed_warcs] = {
      "cld2": {
        "actual_docs": actual_docs_cld2,
        "parallel_docs": parallel_docs_cld2,
      },
      "classifier": {
        "actual_docs": actual_docs_classifier,
        "parallel_docs": parallel_docs_classifier,
      },
      "none": {
        "actual_docs": actual_docs_none,
        "parallel_docs": parallel_docs_none,
      },
    }

warc2websites = {k: sorted(list(v)) for k, v in warc2websites.items()}
websites = sorted(list(websites))
final_data = {}

processed_docs = batch_size
last_n_websites_available = None
processed_docs_individually = {
    "cld2": 0,
    "classifier": 0,
    "none": 0,
}

for processed_warcs in sorted(all_processed_warcs, reverse=False):
  available_n_websites = len(warc2websites[processed_warcs])
  print_last_n_websites_available = False

  if last_n_websites_available is None or last_n_websites_available != available_n_websites:
    print_last_n_websites_available = True

  if print_last_n_websites_available:
    sys.stderr.write(f"Processed docs, warcs and websites: {processed_docs} - {processed_warcs} - {available_n_websites} (previous: {last_n_websites_available}): {' | '.join(warc2websites[processed_warcs])}\n")

  last_n_websites_available = available_n_websites

  for website in warc2websites[processed_warcs]:
    final_data[processed_docs] = {
      "cld2": data[website][processed_warcs]["cld2"]["parallel_docs"],
      "classifier": data[website][processed_warcs]["classifier"]["parallel_docs"],
      "none": data[website][processed_warcs]["none"]["parallel_docs"],
      "docs_cld2": data[website][processed_warcs]["cld2"]["actual_docs"],
      "docs_classifier": data[website][processed_warcs]["classifier"]["actual_docs"],
      "docs_none": data[website][processed_warcs]["none"]["actual_docs"],
    }

    if processed_docs - batch_size > 0:
      final_data[processed_docs]["cld2"] += final_data[processed_docs - batch_size]["cld2"]
      final_data[processed_docs]["classifier"] += final_data[processed_docs - batch_size]["classifier"]
      final_data[processed_docs]["none"] += final_data[processed_docs - batch_size]["none"]
      final_data[processed_docs]["docs_cld2"] += final_data[processed_docs - batch_size]["docs_cld2"]
      final_data[processed_docs]["docs_classifier"] += final_data[processed_docs - batch_size]["docs_classifier"]
      final_data[processed_docs]["docs_none"] += final_data[processed_docs - batch_size]["docs_none"]

    if processed_warcs - batch_size > 0:
      # Subtract previous documents of the previous batch
      final_data[processed_docs]["cld2"] -= data[website][processed_warcs - batch_size]["cld2"]["parallel_docs"]
      final_data[processed_docs]["classifier"] -= data[website][processed_warcs - batch_size]["classifier"]["parallel_docs"]
      final_data[processed_docs]["none"] -= data[website][processed_warcs - batch_size]["none"]["parallel_docs"]
      final_data[processed_docs]["docs_cld2"] -= data[website][processed_warcs - batch_size]["cld2"]["actual_docs"]
      final_data[processed_docs]["docs_classifier"] -= data[website][processed_warcs - batch_size]["classifier"]["actual_docs"]
      final_data[processed_docs]["docs_none"] -= data[website][processed_warcs - batch_size]["none"]["actual_docs"]

    processed_docs += batch_size

print("warcs\tdownloaded documents\tcld2_docs\tclassifier_docs\tnone_docs\tcld2\tclassifier\tnone\tclassifier - cld2\tclassifier - none\tcld2 - none")

for docs in sorted(final_data.keys()):
  cld2 = final_data[docs]["cld2"]
  classifier = final_data[docs]["classifier"]
  none = final_data[docs]["none"]
  docs_cld2 = final_data[docs]["docs_cld2"]
  docs_classifier = final_data[docs]["docs_classifier"]
  docs_none = final_data[docs]["docs_none"]

  print(f"-\t{docs}\t{docs_cld2}\t{docs_classifier}\t{docs_none}\t{cld2}\t{classifier}\t{none}\t{classifier - cld2}\t{classifier - none}\t{cld2 - none}")
