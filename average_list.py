import numpy as np
import skimage.measure

class EverageList:
    def __init__(self, len, dtype=np.float32):
        self.data = np.zeros([len], dtype=np.float32)
        self.data_idx = 0
        self.data_len = len
        self.buffer = np.zeros([1], dtype=np.float32)
        self.buffer_idx = 0
        self.buffer_len = 1
        self.n_data = 0

    def push(self, num):
        self.n_data += 1
        self.buffer[self.buffer_idx] = num
        self.buffer_idx += 1

        if self.buffer_idx == self.buffer_len:
            self.buffer_idx = 0
            self.data[self.data_idx] = np.mean(self.buffer)
            self.data_idx += 1

            if self.data_idx == self.data_len:
                self.data_idx //= 2
                self.data_idx - 1
                self.data[:self.data_len // 2] = skimage.measure.block_reduce(self.data, 2, np.mean)
                self.data[self.data_len // 2:] = 0
                self.buffer_len *= 2
                self.buffer = np.zeros([self.buffer_len], dtype=np.float32)
    
    def get_data(self):
        return self.data[:self.data_idx]

    def get_range(self):
        return range(0, self.n_data + self.buffer_len, self.buffer_len)[:self.data_idx]

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from random import uniform
    fig = plt.figure()
    ax = fig.add_subplot(111)
    l = EverageList(1000)

    for i in range(1, 10000):
        l.push(uniform(-1, 1) - (i * 0.001))

        if i % 10 == 0:
            ax.cla()
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Loss")
            ax.plot(l.get_range(), l.get_data(), "r-", linewidth=1)
            plt.draw()
            plt.pause(0.01)
    plt.pause(100)