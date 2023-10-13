import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from model import Unet
from call_back import CustomCallback
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from dataset import Dataset
import tensorflow as tf


batch_size = 10
epochs = 1000
steps_per_epoch = 50
n_steps = 20


opt = Adam(learning_rate=0.000_01)
loss = "MSE"
gpus = tf.config.list_logical_devices("GPU")
strategy = tf.distribute.OneDeviceStrategy(gpus[0])
# strategy = tf.distribute.experimental.CentralStorageStrategy()
# strategy = tf.distribute.MirroredStrategy(devices=gpus, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
print(f"Number of devices: {strategy.num_replicas_in_sync}")

with strategy.scope():
    model = Unet()
    model.compile(loss=loss, optimizer=opt)

model.summary()

if os.path.exists('model.h5'): model.load_weights('model.h5')

initial_epoch = 0

if os.path.exists('epoch.txt'):
    with open("epoch.txt", "r") as f:
        initial_epoch = int(f.read())


c1 = CustomCallback(chackpoint=True)
c2 = ModelCheckpoint(filepath='model.h5', save_best_only=False, save_weights_only=True, save_freq='epoch')

dataset = Dataset(batch_size, n_steps, (5, 100), 900)

model.fit(dataset, epochs=epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch, initial_epoch=initial_epoch, shuffle=True, callbacks=[c1, c2])

