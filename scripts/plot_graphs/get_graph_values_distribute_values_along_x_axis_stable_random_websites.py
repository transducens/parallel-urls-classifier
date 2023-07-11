
import sys
import random

fn = sys.argv[1]
data = {}
websites = set()
all_processed_warcs = set()
warc2websites = {}
websites2processedwarcs = {}
seed = 42

if len(sys.argv) > 2:
  try:
    seed = int(sys.argv[2])
  except:
    seed = None

if seed is None:
  sys.stderr.write("WARNING: seed has not been set: non-reproducible results\n")

random.seed(seed)

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


    if processed_warcs not in warc2websites:
      warc2websites[processed_warcs] = set()

    warc2websites[processed_warcs].add(website)
    websites2processedwarcs[website] = set()

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

warc2websites = {k: random.sample(sorted(list(v)), k=len(v)) for k, v in warc2websites.items()}
websites = sorted(list(websites))
final_data = {}

processed_docs = batch_size

while len(websites) > 0:
  website = random.sample(websites, k=1)[0]
  max_warc = sorted(data[website].keys())[-1]

  if max_warc in websites2processedwarcs[website]:
    # This website has been exhausted
    websites.remove(website)

    continue

  if len(websites2processedwarcs[website]) == 0:
    processed_warcs = batch_size
  else:
    processed_warcs = batch_size + sorted(list(websites2processedwarcs[website]))[-1]

  final_data[processed_docs] = {
    "cld2": data[website][processed_warcs]["cld2"]["parallel_docs"],
    "classifier": data[website][processed_warcs]["classifier"]["parallel_docs"],
    "none": data[website][processed_warcs]["none"]["parallel_docs"],
  }

  if processed_docs - batch_size > 0:
    final_data[processed_docs]["cld2"] += final_data[processed_docs - batch_size]["cld2"]
    final_data[processed_docs]["classifier"] += final_data[processed_docs - batch_size]["classifier"]
    final_data[processed_docs]["none"] += final_data[processed_docs - batch_size]["none"]

  if processed_warcs - batch_size > 0:
    # Subtract previous documents of the previous batch
    final_data[processed_docs]["cld2"] -= data[website][processed_warcs - batch_size]["cld2"]["parallel_docs"]
    final_data[processed_docs]["classifier"] -= data[website][processed_warcs - batch_size]["classifier"]["parallel_docs"]
    final_data[processed_docs]["none"] -= data[website][processed_warcs - batch_size]["none"]["parallel_docs"]

  processed_docs += batch_size

  websites2processedwarcs[website].add(processed_warcs)

print("warcs\tdownloaded documents\tcld2\tclassifier\tnone\tclassifier - cld2\tclassifier - none\tcld2 - none")

for docs in sorted(final_data.keys()):
  cld2 = final_data[docs]["cld2"]
  classifier = final_data[docs]["classifier"]
  none = final_data[docs]["none"]

  print(f"-\t{docs}\t{cld2}\t{classifier}\t{none}\t{classifier - cld2}\t{classifier - none}\t{cld2 - none}")
