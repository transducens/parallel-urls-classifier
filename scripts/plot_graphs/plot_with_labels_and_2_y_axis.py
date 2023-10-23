
import sys

import matplotlib.pyplot as plt

filename = sys.argv[1]
xlabel = sys.argv[2]
ylabel1 = sys.argv[3]
ylabel2 = sys.argv[4]

provided_files = (len(sys.argv) - 3 - 1) // 2

if len(sys.argv) != 7:
  raise Exception("Unexpected provided args")

file1 = sys.argv[5]
file2 = sys.argv[6]

def read_file(f):
  x, y = [], []
  with open(f) as fd:
    for i in fd:
      i = i.rstrip("\r\n").split('\t')

      x.append(float(i[0]))
      y.append(float(i[1]))

  return x, y

xn, yn = [], []

for f in (file1, file2):
  x, y = read_file(f)

  xn.append(x)
  yn.append(y)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(xn[0], yn[0], 'g-')
ax2.plot(xn[1], yn[1], 'b-')

ax1.set_xlabel(xlabel)
ax1.set_ylabel(ylabel1, color='g')
ax2.set_ylabel(ylabel2, color='b')

plt.savefig(filename)
