
import sys

import matplotlib.pyplot as plt

filename = sys.argv[1]
xlabel = sys.argv[2]
ylabel = sys.argv[3]
files = sys.argv[4:]

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

for f, x, y in zip(files, xn, yn):
  plt.plot(x, y, label=f)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15))
plt.xlabel(xlabel)
plt.ylabel(ylabel)

plt.savefig(filename)
