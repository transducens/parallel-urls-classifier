
import sys

import matplotlib.pyplot as plt

filename = sys.argv[1]
xlabel = sys.argv[2]
ylabel = sys.argv[3]

provided_files = (len(sys.argv) - 3 - 1) // 2

if provided_files * 2 + 3 + 1 != len(sys.argv):
  raise Exception("Unexpected provided args")

labels = sys.argv[4 + 0 * provided_files:4 + 1 * provided_files]
files = sys.argv[4 + 1 * provided_files:4 + 2 * provided_files]

def read_file(f):
  x, y = [], []
  with open(f) as fd:
    for i in fd:
      i = i.rstrip("\r\n").split('\t')

      x.append(int(i[0]))
      y.append(int(i[1]))

  return x, y

xn, yn = [], []

for f in files:
  x, y = read_file(f)

  xn.append(x)
  yn.append(y)

for label, x, y in zip(labels, xn, yn):
  plt.plot(x, y, label=label)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15))
plt.xlabel(xlabel)
plt.ylabel(ylabel)

plt.savefig(filename)
