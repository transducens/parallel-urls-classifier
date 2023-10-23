
import sys

import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

filename = sys.argv[1]
xlabel = sys.argv[2]
ylabel = sys.argv[3]
files = sys.argv[4:]

def read_file(f):
  x, y = [], []
  with open(f) as fd:
    for i in fd:
      i = i.rstrip("\r\n").split('\t')

      x.append(float(i[0]))
      y.append(float(i[1]))

  return x, y

xn, yn, linregn = [], [], []

for f in files:
  x, y = read_file(f)
  linreg = stats.linregress(x, y)

  xn.append(x)
  yn.append(y)
  linregn.append(linreg)

tinv = lambda p, df: abs(stats.t.ppf(p/2, df))
ts = tinv(0.05, len(x)-2)

for f, x, y, linreg in zip(files, xn, yn, linregn):
  plt.plot(x, y, 'o', label=f)
  plt.plot(x, linreg.intercept + linreg.slope * np.array(x), label=f"Fitted line: {f}")

  print(f"{f}: R-squared: {linreg.rvalue ** 2:.6f}")
  print(f"{f}: slope (95%): {linreg.slope:.6f} +/- {ts * linreg.stderr:.6f}")
  print(f"{f}: intercept (95%): {linreg.intercept:.6f} +/- {ts * linreg.intercept_stderr:.6f}")

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15))
plt.xlabel(xlabel)
plt.ylabel(ylabel)

plt.savefig(filename)
