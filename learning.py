import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import keras
# from keras import layers
from sempler import Dataset
from wave_u_net import wave_u_net
from loss import combined_loss, ScatterLoss
from call_back import CustomCallback
from keras.callbacks import ModelCheckpoint
# import random
# from matplotlib import pyplot as plt
import json
# import numpy as np
import math

s_size = 16384 * (24 // 2)
steps_per_epoch = 10
steps = 100
noise_ratio = 0.7

# model = wave_u_net(num_initial_filters = 12, num_layers = 6, kernel_size = 10, input_size = s_size, output_type = "single")

# model = wave_u_net(num_initial_filters = 24, num_layers = 12, kernel_size = 15, input_size = s_size, output_type = "single")
# model = wave_u_net(num_initial_filters = 32, num_layers = 16, kernel_size = 30, input_size = s_size, output_type = "single")
# model = wave_u_net(num_initial_filters = 32, num_layers = 16, kernel_size = 50, input_size = s_size, output_type = "single")

model = wave_u_net(num_initial_filters = 24, num_layers = 12, kernel_size = 15, input_size = s_size, output_type = "single", attention = "Polarized", dropout = True, dropout_rate = 0.2)
# model = wave_u_net(num_initial_filters = 32, num_layers = 16, kernel_size = 30, input_size = s_size, output_type = "single", attention = "Polarized", dropout = True, dropout_rate = 0.2)
# model = wave_u_net(num_initial_filters = 32, num_layers = 16, kernel_size = 50, input_size = s_size, output_type = "single", attention = "Polarized", dropout = True, dropout_rate = 0.2)

if os.path.exists('model.h5'): model.load_weights('model.h5')

initial_epoch = 0

if os.path.exists('epoch.h5'):
    with open("epoch.txt", "r") as f:
        initial_epoch = int(f.read())

opt = keras.optimizers.Adam(learning_rate=0.000_01) #0.000_01

loss = ScatterLoss(s_size, steps, noise_ratio)
# loss = combined_loss

model.compile(loss="MSE", optimizer=opt)

model.summary()

dataset = Dataset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], s_size=s_size, steps=steps, batch_size=1, noise_ratio=noise_ratio)

epochs = len(dataset) // steps_per_epoch

history = model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, initial_epoch=initial_epoch, shuffle=False, callbacks=[CustomCallback(chackpoint=True), ModelCheckpoint(filepath='model.h5', save_best_only=False, save_weights_only=True, save_freq='epoch')])

