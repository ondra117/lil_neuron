from scipy.io import wavfile
import numpy as np
from keras.utils import Sequence
from threading import Thread, Lock
from time import sleep
from random import shuffle, seed

# def _get_sample(idx, s_size, data, movs, steps, noise_ratio, orig=False):

#     sound = data[idx * movs:idx * movs + s_size]
#     first_noise = np.random.normal(0.0, 1, size=s_size)
#     noise = first_noise[:]

#     for d in range(1, steps):
#         noise[d * movs:] += np.random.normal(0.0, 1, size=s_size - d * movs)

#     norm = np.abs(noise).max()
#     noise /= norm

#     samples = noise * (noise_ratio) + sound * (1 - noise_ratio)

#     if orig:
#         out = sound
#     else:
#         out = first_noise / norm
#         out *= noise_ratio

#     return samples.reshape([-1, 1]), out.reshape([-1, 1])


def _get_sample(idx, s_size, data, movs, steps, noise_ratio, orig=False):

    sound = data[idx * movs:idx * movs + s_size]
    noise = np.random.normal(0.0, 1, size=s_size)
    noise /= np.abs(noise).max()

    samples = np.zeros_like(sound)
    # print("start")
    for d in range(steps):
        start = d * movs
        end = start + movs
        noise_volium = (d + 1) / steps
        samples[start:end] = sound[start:end] * (1 - noise_volium) + noise[start:end] * noise_volium
    #     print(f"d:{d} | nv:{noise_volium} | dip:{np.mean(np.abs(samples[start:end] - sound[start:end]))} | dima:{np.max(np.abs(samples[start:end] - sound[start:end]))} | dimi:{np.min(np.abs(samples[start:end] - sound[start:end]))}")
    # print("end")
    if orig:
        out = sound
    else:
        out = noise

    return samples.reshape([-1, 1]), out.reshape([-1, 1])


class Dataset(Sequence):
    def __init__(self, songs, s_size=16384 * 24, steps=10, batch_size=1, noise_ratio=0.5, random_seed=0, orig=False, info=False):
        np.random.seed(random_seed)
        self.s_size = s_size
        self.steps = steps
        self.batch_size = batch_size
        self.noise_ratio = noise_ratio
        self.orig = orig
        self.movs = s_size // steps
        self.idx = 0
        self.X = np.empty([batch_size, s_size, 1], dtype=np.float32)
        self.Y = np.empty([batch_size, s_size, 1], dtype=np.float32)
        self.lock = Lock()
        self.generate = False

        norm = lambda x: x / np.abs(x).max()

        if info:
            n_songs = len(songs)
            self.songs = [(wavfile.read(f'./music/{song}.wav')[1].astype(np.float32), print(f"Loading songs: {idx}/{n_songs}", end="\r"))[0] for idx, song in enumerate(songs)]
            print(f"Loading songs: {n_songs}/{n_songs}")
        else:
            self.songs = [wavfile.read(f'./music/{song}.wav')[1].astype(np.float32) for song in songs]

        for idx, song in enumerate(self.songs):
            if song.ndim == 2:
                song = np.mean(song, axis=1)
            self.songs[idx] = norm(song)
            if info:
                print(f"Processing songs: {idx}/{n_songs}", end="\r")
                
        if info:
            print(f"Processing songs: {n_songs}/{n_songs}")

        self.songs = np.concatenate(self.songs)
        
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
                    self.X[idx], self.Y[idx] = _get_sample(start * self.batch_size + idx, self.s_size, self.songs, self.movs, self.steps, self.noise_ratio, orig=self.orig)
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
    
def _get_sample_side(idx, s_size, data, movs, steps, noise_ratio, cycles, orig=False):

    s_idx = idx[1] * movs
    sound = data[idx[0]][s_idx:s_idx + s_size]
    side = np.zeros([s_size * cycles])
    side[-s_idx:] = data[idx[0]][max(0, s_idx-side.size):s_idx]
    noise = np.random.normal(0.0, 1, size=s_size)
    noise /= np.abs(noise).max()

    samples = np.zeros_like(sound)
    # print("start")
    for d in range(steps):
        start = d * movs
        end = start + movs
        noise_volium = (d + 1) / steps
        samples[start:end] = sound[start:end] * (1 - noise_volium) + noise[start:end] * noise_volium
    #     print(f"d:{d} | nv:{noise_volium} | dip:{np.mean(np.abs(samples[start:end] - sound[start:end]))} | dima:{np.max(np.abs(samples[start:end] - sound[start:end]))} | dimi:{np.min(np.abs(samples[start:end] - sound[start:end]))}")
    # print("end")
    if orig:
        out = sound
    else:
        out = noise

    return samples.reshape([-1, 1]), side.reshape([s_size, cycles]), out.reshape([-1, 1])
    

class DatasetSide(Sequence):
    def __init__(self, songs, s_size=16384 * 24, steps=10, batch_size=1, noise_ratio=0.5, random_seed=0, orig=False, info=False, side_cysles=10):
        seed(random_seed)
        self.s_size = s_size
        self.steps = steps
        self.batch_size = batch_size
        self.noise_ratio = noise_ratio
        self.orig = orig
        self.side_cycles = side_cysles
        self.movs = s_size // steps
        self.idx = 0
        self.X = [np.empty([batch_size, s_size, 1], dtype=np.float32), np.empty([batch_size, s_size, side_cysles], dtype=np.float32)]
        self.Y = np.empty([batch_size, s_size, 1], dtype=np.float32)
        self.lock = Lock()
        self.generate = False

        norm = lambda x: x / np.abs(x).max()

        if info:
            n_songs = len(songs)
            self.songs = [(wavfile.read(f'./music/{song}.wav')[1].astype(np.float32), print(f"Loading songs: {idx}/{n_songs}", end="\r"))[0] for idx, song in enumerate(songs)]
            print(f"Loading songs: {n_songs}/{n_songs}")
        else:
            self.songs = [wavfile.read(f'./music/{song}.wav')[1].astype(np.float32) for song in songs]

        for idx, song in enumerate(self.songs):
            if song.ndim == 2:
                song = np.mean(song, axis=1)
            self.songs[idx] = norm(song)
            if info:
                print(f"Processing songs: {idx}/{n_songs}", end="\r")
                
        if info:
            print(f"Processing songs: {n_songs}/{n_songs}")

        # self.songs = np.concatenate(self.songs)
        
        songs_composition = [((song.shape[0]) // self.movs) // batch_size for song in self.songs]
        self.data_len = sum(songs_composition)

        self.songs = [np.append(song, np.zeros([s_size * batch_size])) for song in self.songs]

        self.shufle = [(idx, sample) for idx, song_len in enumerate(songs_composition) for sample in range(song_len)]
        shuffle(self.shufle)

        Thread(target=self._generate_new).start()

    def __len__(self):
        return self.data_len

    def _generate_new(self):
        while True:
            if not self.generate:
                start = self.shufle[self.idx * self.batch_size]
                for idx in range(self.batch_size):
                    self.X[0][idx], self.X[1][idx],self.Y[idx] = _get_sample_side((start[0], start[1] + idx), self.s_size, self.songs, self.movs, self.steps, self.noise_ratio, self.side_cycles, orig=self.orig)
                self.generate = True
            sleep(0.00001)

    def __getitem__(self, idx):
        # start = idx * self.batch_size
        # for idx in range(self.batch_size):
        #     self.X[idx], self.Y[idx] = _get_sample(start + idx, self.s_size, self.songs, self.movs, self.steps, self.noise_ratio)
        # return self.X, self.Y
        while not self.generate:
            sleep(0.00001)
        self.idx = idx
        X = self.X
        Y = self.Y
        self.lock.acquire()
        self.generate = False
        self.lock.release()
        return X, Y
    
def _get_sample_soft(idx, s_size, data, movs, steps, noise_ratio, orig=False):

    sound = data[idx * movs:idx * movs + s_size]
    noise = np.random.normal(0.0, 1, size=s_size)
    noise /= np.abs(noise).max()

    samples = np.zeros_like(sound)
    # print("start")
    for d in range(steps):
        start = d * movs
        end = start + movs
        noise_volium = (d + 1) / steps
        samples[start:end] = sound[start:end] * (1 - noise_volium) + noise[start:end] * noise_volium
    #     print(f"d:{d} | nv:{noise_volium} | dip:{np.mean(np.abs(samples[start:end] - sound[start:end]))} | dima:{np.max(np.abs(samples[start:end] - sound[start:end]))} | dimi:{np.min(np.abs(samples[start:end] - sound[start:end]))}")
    # print("end")
    if orig:
        out = sound
    else:
        out = noise

    return samples.reshape([-1, 1]), out.reshape([-1, 1])

class DatasetSoft(Sequence):
    def __init__(self, songs, s_size=16384 * 24, steps=10, batch_size=1, noise_ratio=0.5, random_seed=0, orig=False, info=False):
        seed(random_seed)
        self.s_size = s_size
        self.steps = steps
        self.batch_size = batch_size
        self.noise_ratio = noise_ratio
        self.orig = orig
        self.movs = s_size // steps
        self.idx = 0
        self.X = np.empty([batch_size, s_size, 1], dtype=np.float32)
        self.Y = np.empty([batch_size, s_size, 1], dtype=np.float32)
        self.lock = Lock()
        self.generate = False

        n_songs = len(songs)
        songs_len = [None] * n_songs
        for idx, song in enumerate(songs):
            songs_len[idx] = wavfile.read(f'./music/{song}.wav')[1].astype(np.float32).shape[0]
            if info:
                print(f"Loading songs: {idx}/{n_songs}", end="\r")
        if info:
            print(f"Loading songs: {n_songs}/{n_songs}")

        songs_composition = [((song) // self.movs) // batch_size for song in songs_len]
        self.data_len = sum(songs_composition)

        self.shufle = [(idx, sample) for idx, song_len in enumerate(songs_composition) for sample in range(song_len)]
        shuffle(self.shufle)

        Thread(target=self._generate_new).start()

    def __len__(self):
        return self.data_len

    def _generate_new(self):
        while True:
            if not self.generate:
                start = self.shufle[self.idx * self.batch_size]
                data = wavfile.read(f'./music/{start[0]}.wav')[1].astype(np.float32)
                if data.ndim == 2:
                    data = np.mean(data, axis=1)
                data /= np.abs(data).max()
                data = np.append(data, np.zeros([self.s_size * self.batch_size]))
                for idx in range(self.batch_size):
                    self.X[idx], self.Y[idx] = _get_sample_soft(start[1] * self.batch_size + idx, self.s_size, data, self.movs, self.steps, self.noise_ratio, orig=self.orig)
                self.generate = True
            sleep(0.00001)

    def __getitem__(self, idx):
        while not self.generate:
            sleep(0.00001)
        self.idx = idx
        X = self.X
        Y = self.Y
        self.lock.acquire()
        self.generate = False
        self.lock.release()
        return X, Y


from time import time, sleep
if __name__ == '__main__':
    t = time()
    dataset = DatasetSoft([0, 1], s_size=16384 * 12, steps=40, batch_size=3, noise_ratio=0.7, info=True)
    for i in range(10):
        # sleep(1)
        t = time()
        dataset[0]
        print(time() - t)
    # wavfile.write(f"t.wav", 44000, (dataset[0][0].reshape([-1]) * 32767).astype(np.int16))
    print("done")
    exit()
    
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
