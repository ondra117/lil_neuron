import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow import keras
from keras import layers
from scipy.io import wavfile
import numpy as np
from wave_u_net import wave_u_net
from time import time
import random
from sempler import Dataset
import datetime

s_size = 16384 * (24 // 2)
# steps_per_epoch = 10
side = False
cycles = 10
steps = 10

# model = wave_u_net(num_initial_filters = 24, num_layers = 12, kernel_size = 15, input_size = s_size, output_type = "single")
# model = wave_u_net(num_initial_filters = 32, num_layers = 16, kernel_size = 30, input_size = s_size, output_type = "single")
# model = wave_u_net(num_initial_filters = 32, num_layers = 16, kernel_size = 50, input_size = s_size, output_type = "single")

# model = wave_u_net(num_initial_filters = 24, num_layers = 12, kernel_size = 15, input_size = s_size, output_type = "single", attention = "Gate", attention_res = False, dropout = "False", dropout_rate = 0.2)
# model = wave_u_net(num_initial_filters = 32, num_layers = 16, kernel_size = 30, input_size = s_size, output_type = "single", attention = "Gate", attention_res = False, dropout = "False", dropout_rate = 0.2, sub=False, side_chanel=side, side_chanel_cycles=cycles)
model = wave_u_net(num_initial_filters = 32, num_layers = 16, kernel_size = 50, input_size = s_size, output_type = "single", attention = "Gate", attention_res = False, dropout = False, dropout_rate = 0.2, sub=False, side_chanel=side, side_chanel_cycles=cycles)

model.load_weights("model.h5")

model.summary()

dataset = Dataset([1], s_size=s_size, steps=steps, batch_size=1, noise_ratio=0.7)

sound = dataset.songs
noise = np.random.normal(0.0, 1, size=sound.shape[0])

noise = noise / np.max(np.abs(noise))

sound = (noise * (dataset.noise_ratio) + sound * (1 - dataset.noise_ratio))

# wavfile.write("o.wav", 44000, (sound * 32767 * 0.5).astype(np.int16))

sound = sound.reshape([1, -1, 1])
if side:
    side_input = np.zeros([1, s_size, cycles])

print(sound.shape[1])
#
t = 0
gt = 0
i = 0
while i * dataset.movs + s_size < sound.shape[1]:
    print(f"{i * dataset.movs + s_size} / {sound.shape[1]} | ETA: {datetime.timedelta(seconds=(sound.shape[1] - (i * dataset.movs + s_size)) / dataset.movs * gt)}", end="\r")
    t = time()
    if side:
        s_idx = i * dataset.movs
        side_input.fill(0)
        side_input = side_input.reshape([-1])
        if not s_idx == 0:
            side_input[-s_idx:] = sound.reshape([-1])[max(0, s_idx-side_input.size):s_idx]
        side_input = side_input.reshape([1, s_size, cycles])
        noise = model.predict([sound[:, s_idx:i * dataset.movs + s_size, :], side_input], verbose = 0)
    else:
        noise = model.predict(sound[:, i * dataset.movs:i * dataset.movs + s_size, :], verbose = 0)
    sound[:, i * dataset.movs:i * dataset.movs + s_size, :] -= noise
    t = time() - t
    gt += t
    gt /= 2
    i += 1


sound = np.array(sound).reshape([-1])

wavfile.write("t.wav", 44000, (sound * 32767).astype(np.int16))
print("done" + " " * 50)