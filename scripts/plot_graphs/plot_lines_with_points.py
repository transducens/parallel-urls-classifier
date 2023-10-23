
import sys

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.xmargin'] = 0

title = sys.argv[1]
xlabel = sys.argv[2]
ylabel = sys.argv[3].split(':')
output = sys.argv[4]
bar_labels = sys.argv[5].split(':')
points_labels = sys.argv[6].split(':')

data = {}
langs = []

for l in sys.stdin:
  l = l.rstrip("\r\n").split('\t')
  lang = l[0]

  if lang not in data:
    data[lang] = {"bar": None, "points": None}
    data[lang]["bar"] = {k: float(l[idx]) for idx, k in enumerate(bar_labels, 1)}
    data[lang]["points"] = {k: float(l[idx]) for idx, k in enumerate(points_labels, 1 + len(bar_labels))}
    langs.append(lang)
  else:
    raise Exception("Error: again")

data2 = {"bar": {}, "points": {}}

for bar_label in bar_labels:
  data2["bar"][bar_label] = [data[lang]["bar"][bar_label] for lang in langs]

for points_label in points_labels:
  data2["points"][points_label] = [data[lang]["points"][points_label] for lang in langs]

data = data2

# data to plot
n_groups = len(langs) # langs

# create plot
fig, ax1 = plt.subplots()
index = np.arange(n_groups)
index = index.astype(np.float64)
bar_width = 0.5
opacity = 0.8

ax2 = ax1.twinx()

for i in range(len(index)):
  index[i] += i * bar_width

#ax.set_xticklabels(x_labels, rotation=90, va="center")
fig.set_size_inches(30, 6)

plots = []

for idx, bar_label in enumerate(bar_labels):
  p = ax1.plot(index + bar_width * len(bar_labels) / 2, data["bar"][bar_label], 'o-', label=bar_label)

  plots.extend(p)

cmap = plt.get_cmap("Dark2")

for idx, points_label in enumerate(points_labels):
  p = ax2.plot(index + bar_width * len(bar_labels) / 2, data["points"][points_label], 'o', markeredgecolor="black", label=points_label, color=cmap(idx))

  plots.extend(p)

ax1.set_xlabel(xlabel)
ax1.set_ylabel(ylabel[0])
ax2.set_ylabel(ylabel[1])
plt.title(title)
#plt.xticks(index + bar_width, langs, rotation=90, va="center", position=(0, -0.01))
ax1.set_xticks(index + bar_width, langs, rotation="vertical")

lns = plots
labs = [l.get_label() for l in plots]
ax1.legend(lns, labs)

#ax1.legend()

plt.tight_layout()

plt.savefig(output)
