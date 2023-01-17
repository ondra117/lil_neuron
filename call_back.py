import os
from keras.callbacks import Callback
from average_list import EverageList
from matplotlib import pyplot as plt
import pickle

class CustomCallback(Callback):
    def __init__(self, chackpoint=False, file_name="chackpoint"):
        self.chackpoint = chackpoint
        self.file_name = file_name + ".pkl"
        if os.path.exists(self.file_name) and chackpoint:
            with open(self.file_name, 'rb') as f:
                data = f.read()
            self.loss_data = pickle.loads(data)
        else:
            self.loss_data = EverageList(10000)
        # fig = plt.figure()
        # self.ax = fig.add_subplot(111)
        super().__init__()

    def on_batch_end(self, epoch, logs=None):
        self.loss_data.push(logs['loss'])

    def on_epoch_end(self, epoch, logs=None):
        data = pickle.dumps(self.loss_data)
        with open(self.file_name, 'wb') as f:
            f.write(data)

        # self.ax.cla()
        # # self.ax.set_xscale('log')
        # self.ax.set_yscale('log')
        # self.ax.set_xlabel("Iterations")
        # self.ax.set_ylabel("Loss")
        # self.ax.plot(self.loss_data.get_range(), self.loss_data.get_data(), "r-", linewidth=0.2)
        # plt.draw()
        # plt.pause(0.01)

    