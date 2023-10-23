
import sys

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.xmargin'] = 0

title = sys.argv[1]
xlabel = sys.argv[2]
ylabel = sys.argv[3]
output = sys.argv[4]
bar_labels = sys.argv[5:]

data = {}
langs = []

for l in sys.stdin:
  l = l.rstrip("\r\n").split('\t')
  lang = l[0]

  if len(bar_labels) != len(l) - 1:
    raise Exception("Error")

  if lang not in data:
    data[lang] = {k: float(l[idx]) for idx, k in enumerate(bar_labels, 1)}
    langs.append(lang)
  else:
    raise Exception("Error: again")

data2 = {}

for bar_label in bar_labels:
  data2[bar_label] = [data[lang][bar_label] for lang in langs]

data = data2

# data to plot
n_groups = len(langs) # langs

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
index = index.astype(np.float64)
bar_width = 0.5
opacity = 0.8

for i in range(len(index)):
  index[i] += i * bar_width

#ax.set_xticklabels(x_labels, rotation=90, va="center")
fig.set_size_inches(30, 6)

for idx, bar_label in enumerate(bar_labels):
  plt.bar(index + bar_width * idx, data[bar_label], align='edge', width=bar_width, alpha=opacity, label=bar_label)

plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
#plt.xticks(index + bar_width, langs, rotation=90, va="center", position=(0, -0.01))
plt.xticks(index + bar_width, langs, rotation="vertical")
plt.legend()

plt.tight_layout()

plt.savefig(output)
