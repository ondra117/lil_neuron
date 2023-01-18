from scipy.io import wavfile
import numpy as np
from keras.utils import Sequence
from threading import Thread, Lock
from time import sleep

def _get_sample(idx, s_size, data, movs, steps, noise_ratio):

    sound = data[idx * movs:idx * movs + s_size]
    noise = np.random.normal(0.0, 1, size=s_size)

    for d in range(1, steps):
        noise[d * movs:] += np.random.normal(0.0, 1, size=s_size - d * movs)

    noise /= np.abs(noise).max()

    samples = noise * (noise_ratio) + sound * (1 - noise_ratio)

    return samples.reshape([-1, 1]), noise.reshape([-1, 1])

class Dataset(Sequence):
    def __init__(self, songs, s_size=16384 * 24, steps=10, batch_size=1, noise_ratio=0.5, random_seed=0):
        np.random.seed(random_seed)
        self.s_size = s_size
        self.steps = steps
        self.batch_size = batch_size
        self.noise_ratio = noise_ratio
        self.movs = s_size // steps
        self.idx = 0
        self.X = np.empty([batch_size, s_size, 1], dtype=np.float32)
        self.Y = np.empty([batch_size, s_size, 1], dtype=np.float32)
        self.lock = Lock()
        self.generate = False

        norm = lambda x: x / np.abs(x).max()

        self.songs = np.concatenate([norm(np.mean(wavfile.read(f'./music/{song}.wav')[1], axis=1).astype(np.float32)) for song in songs])

        self.data_len = ((self.songs.shape[0]) // self.movs) // batch_size

        self.songs = np.append(self.songs, np.zeros([s_size * batch_size]))

        self.shufle = np.arange(self.data_len)
        np.random.shuffle(self.shufle)

        Thread(target=self._generate_new).start()

    def __len__(self):
        return self.data_len

    def _generate_new(self):
        while True:
            if not self.generate:
                start = self.shufle[self.idx * self.batch_size]
                for idx in range(self.batch_size):
                    self.X[idx], self.Y[idx] = _get_sample(start + idx, self.s_size, self.songs, self.movs, self.steps, self.noise_ratio)
                self.generate = True
            sleep(0.00001)

    def __getitem__(self, idx):
        # start = idx * self.batch_size
        # for idx in range(self.batch_size):
        #     self.X[idx], self.Y[idx] = _get_sample(start + idx, self.s_size, self.songs, self.movs, self.steps, self.noise_ratio)
        # return self.X, self.Y
        while not self.generate:
            sleep(0.00001)
        X = self.X
        Y = self.Y
        self.lock.acquire()
        self.generate = False
        self.lock.release()
        return X, Y

from time import time, sleep
if __name__ == '__main__':
    t = time()
    dataset = Dataset([0, 1], s_size=16384 * 12, steps=100, batch_size=1, noise_ratio=0.7)
    for i in range(10):
        sleep(1)
        t = time()
        dataset[0]
        print(time() - t)
    exit()
    # wavfile.write(f"t.wav", 44000, (dataset[5000][0].reshape([-1]) * 32767).astype(np.int16))
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)

    sample = dataset[500][0].reshape([-1])
    noise = dataset[500][1].reshape([-1])

    ax.plot(np.arange(sample.shape[0]), sample,'r-', linewidth=0.1)
    ax.plot(np.arange(sample.shape[0]), sample-noise,'g-', linewidth=0.1)

    plt.show()
    
    
    # print(time() - t)
    # print(data[0].shape)

    # Sxx = data[0][2]

    # print(np.max(Sxx))
    # print(np.min(Sxx))
