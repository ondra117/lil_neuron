import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import keras
# from keras import layers
from sempler import Dataset, DatasetSide, DatasetSoft
from wave_u_net import wave_u_net
from loss import combined_loss, ScatterLoss, RegulatedLoss
from call_back import CustomCallback
from keras.callbacks import ModelCheckpoint
# import random
# from matplotlib import pyplot as plt
import json
# import numpy as np
import math

s_size = 16384 * (24 // 2)
steps_per_epoch = 100
steps = 10 # 40
noise_ratio = 0.7
batch_size=3
side = False
cycles = 10

# model = wave_u_net(num_initial_filters = 12, num_layers = 6, kernel_size = 10, input_size = s_size, output_type = "single")

# model = wave_u_net(num_initial_filters = 24, num_layers = 12, kernel_size = 15, input_size = s_size, output_type = "single")
# model = wave_u_net(num_initial_filters = 32, num_layers = 16, kernel_size = 30, input_size = s_size, output_type = "single")
# model = wave_u_net(num_initial_filters = 32, num_layers = 16, kernel_size = 50, input_size = s_size, output_type = "single")

# model = wave_u_net(num_initial_filters = 24, num_layers = 12, kernel_size = 15, input_size = s_size, output_type = "single", attention = "Gate", attention_res = False, dropout = "False", dropout_rate = 0.2, sub=True, side_chanel=True, side_chanel_cycles=10)
# model = wave_u_net(num_initial_filters = 32, num_layers = 16, kernel_size = 30, input_size = s_size, output_type = "single", attention = "Gate", attention_res = False, dropout = "False", dropout_rate = 0.2, sub=True, side_chanel=side, side_chanel_cycles=cycles)
model = wave_u_net(num_initial_filters = 32, num_layers = 16, kernel_size = 50, input_size = s_size, output_type = "single", attention = "Gate", attention_res = False, dropout = "Last", dropout_rate = 0.2, sub=True, side_chanel=side, side_chanel_cycles=cycles)

if os.path.exists('model.h5'): model.load_weights('model.h5')

initial_epoch = 0

if os.path.exists('epoch.txt'):
    with open("epoch.txt", "r") as f:
        initial_epoch = int(f.read())

opt = keras.optimizers.Adam(learning_rate=0.000_01) #0.000_1 - 0.000_01

loss = RegulatedLoss(s_size, steps, noise_ratio)
# loss = ScatterLoss(s_size, steps, noise_ratio)
# loss = "MSE"

model.compile(loss=loss, optimizer=opt)

model.summary()

c1 = CustomCallback(chackpoint=True)
c2 = ModelCheckpoint(filepath='model.h5', save_best_only=False, save_weights_only=True, save_freq='epoch')

# 90
if side:
    dataset = DatasetSide(list(range(90)), s_size=s_size, steps=steps, batch_size=batch_size, noise_ratio=noise_ratio, orig=True, info=True, side_cysles=cycles)
else:
    # dataset = Dataset(list(range(900, 1044)), s_size=s_size, steps=steps, batch_size=batch_size, noise_ratio=noise_ratio, orig=True, info=True)
    dataset = DatasetSoft(list(range(0, 200)), s_size=s_size, steps=steps, batch_size=batch_size, noise_ratio=noise_ratio, orig=True, info=True)

epochs = len(dataset) // steps_per_epoch
print(f"data: {(len(dataset) * batch_size):_}")

model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, initial_epoch=initial_epoch, shuffle=False, callbacks=[c1, c2])

