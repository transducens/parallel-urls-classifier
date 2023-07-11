
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
max_values = {website: {"cld2": 0, "classifier": 0, "none": 0} for website in websites}

for website in websites:
  max_processed_warcs = sorted(data[website].keys())[-1]

  max_values[website]["cld2"] = data[website][max_processed_warcs]["cld2"]["parallel_docs"]
  max_values[website]["classifier"] = data[website][max_processed_warcs]["classifier"]["parallel_docs"]
  max_values[website]["none"] = data[website][max_processed_warcs]["none"]["parallel_docs"]

final_data = {}
found_websites = set()

for processed_warcs in sorted(all_processed_warcs, reverse=True):
  if processed_warcs not in final_data:
    final_data[processed_warcs] = {"cld2": 0, "classifier": 0, "none": 0}

  for website in websites:
    if processed_warcs in data[website]:
      found_websites.add(website)

      final_data[processed_warcs]["cld2"] += data[website][processed_warcs]["cld2"]["parallel_docs"]
      final_data[processed_warcs]["classifier"] += data[website][processed_warcs]["classifier"]["parallel_docs"]
      final_data[processed_warcs]["none"] += data[website][processed_warcs]["none"]["parallel_docs"]

    elif website not in found_websites:
      final_data[processed_warcs]["cld2"] += max_values[website]["cld2"]
      final_data[processed_warcs]["classifier"] += max_values[website]["classifier"]
      final_data[processed_warcs]["none"] += max_values[website]["none"]

print("warcs\tdocuments per website\tcld2\tclassifier\tnone\tclassifier - cld2\tclassifier - none\tcld2 - none")

for processed_warcs in sorted(all_processed_warcs):
  cld2 = final_data[processed_warcs]["cld2"]
  classifier = final_data[processed_warcs]["classifier"]
  none = final_data[processed_warcs]["none"]

  print(f"{processed_warcs}\t{processed_warcs}\t{cld2}\t{classifier}\t{none}\t{classifier - cld2}\t{classifier - none}\t{cld2 - none}")
