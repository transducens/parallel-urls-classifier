
import sys

fn = sys.argv[1]
data = {}
websites = set()
all_processed_warcs = set()

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

    if actual_docs_cld2 % batch_size != 0 or actual_docs_classifier % batch_size != 0 or actual_docs_none % batch_size != 0:
      continue

    websites.add(website)
    all_processed_warcs.add(processed_warcs)

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

websites = sorted(list(websites))
final_data = {}
last_processed_warcs = None
website2lastwarc = {}

for processed_warcs in sorted(all_processed_warcs, reverse=False):
  if processed_warcs not in final_data:
    final_data[processed_warcs] = {"cld2": 0, "classifier": 0, "none": 0, "sum_docs": 0}

    if last_processed_warcs is not None:
      final_data[processed_warcs]["sum_docs"] += final_data[last_processed_warcs]["sum_docs"]

  n_websites = 0

  for website in websites:
    if processed_warcs in data[website]:
      final_data[processed_warcs]["cld2"] += data[website][processed_warcs]["cld2"]["parallel_docs"]
      final_data[processed_warcs]["classifier"] += data[website][processed_warcs]["classifier"]["parallel_docs"]
      final_data[processed_warcs]["none"] += data[website][processed_warcs]["none"]["parallel_docs"]
      n_websites += 1
      website2lastwarc[website] = processed_warcs
    elif website in website2lastwarc:
      final_data[processed_warcs]["cld2"] += data[website][website2lastwarc[website]]["cld2"]["parallel_docs"]
      final_data[processed_warcs]["classifier"] += data[website][website2lastwarc[website]]["classifier"]["parallel_docs"]
      final_data[processed_warcs]["none"] += data[website][website2lastwarc[website]]["none"]["parallel_docs"]
    else:
      sys.stderr.write("ERROR: what?\n")

  final_data[processed_warcs]["sum_docs"] += batch_size * n_websites

  last_processed_warcs = processed_warcs

print("warcs\tdownloaded documents\tcld2\tclassifier\tnone\tclassifier - cld2\tclassifier - none\tcld2 - none")

for processed_warcs in sorted(all_processed_warcs):
  cld2 = final_data[processed_warcs]["cld2"]
  classifier = final_data[processed_warcs]["classifier"]
  none = final_data[processed_warcs]["none"]
  docs = final_data[processed_warcs]["sum_docs"]

  print(f"{processed_warcs}\t{docs}\t{cld2}\t{classifier}\t{none}\t{classifier - cld2}\t{classifier - none}\t{cld2 - none}")
