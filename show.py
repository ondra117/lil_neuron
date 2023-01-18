from matplotlib import pyplot as plt
import pickle
import numpy as np

file_name = "chackpoint.pkl"
base = 2

with open(file_name, "rb") as f:
    data = f.read()

loss_data = pickle.loads(data)

# lhm = np.convolve(np.array(loss, dtype=np.float16), np.full([100], 0.01, dtype=np.float16))
# ltm = np.convolve(np.array(loss, dtype=np.float16), np.full([1000], 0.001, dtype=np.float16))

# plt.set_xlabel("Iterations")
# plt.set_ylabel("Loss")

fig = plt.figure()
ax = fig.add_subplot(111)

data = loss_data.get_data()

# data = np.log10(data) + 1

ax.set_yscale('log')

print(data[-1])
ax.plot(loss_data.get_range(), data, "r-", linewidth=0.2)
# ax.plot(range(len(lhm)), lhm, "b-", linewidth=1)
# ax.plot(range(len(ltm)), ltm, "y-", linewidth=1)
# ax.plot(cicles_ends[0], cicles_ends[1], "g.")
plt.show()