from keras.utils import Sequence
from scipy.io import wavfile
import numpy as np
from random import randint
from sampler import Sampler

class Dataset(Sequence):
    def __init__(self, batch_size, n_timesteps, text_embed_dim, file_val, input_size=512):
        self.batch_size = batch_size
        self.n_timesteps = n_timesteps
        self.text_embed_dim = text_embed_dim
        self.file_val = file_val + 1
        self.input_size=input_size
        self.nouse_sheduler = Sampler.cosine(self.n_timesteps, 0)

    def __len__(self):
        return 1000000000

    def __getitem__(self, _):
        idxs = np.random.randint(0, self.file_val, size=self.batch_size)
        sounds = [wavfile.read(f"music/{str(idx).zfill(5)}.wav")[1].astype(np.float32) / 32767 for idx in idxs]
        
        sounds_samples = [randint(self.input_size, len(sound) - self.input_size) for sound in sounds]
        samples = [sound[sound_sample:sound_sample + self.input_size] for sound, sound_sample in zip(sounds, sounds_samples)]
        X_time = np.random.randint(1, self.n_timesteps, size=self.batch_size)
        Y = np.empty((self.batch_size, self.input_size), dtype=np.float32)
        X_samples = np.empty((self.batch_size, self.input_size), dtype=np.float32)
        for idx, (sample, t) in enumerate(zip(samples, X_time)):
            X_samples[idx], Y[idx] = self.nouse_sheduler.q_sample(sample, t)
        X_samples = X_samples[:, :, None]

        X_sound_time = np.array(sounds_samples, dtype=np.float32) / 44000

        X_generated_sound = [sound[:sound_sample][:, None] for sound, sound_sample in zip(sounds, sounds_samples)]

        X_text_embed = np.ones((self.batch_size, *self.text_embed_dim), dtype=np.float32)

        X = [X_samples, X_time, X_sound_time, X_generated_sound, X_text_embed]

        return X, Y

if __name__ == "__main__":
    dataset = Dataset(2, 3, (10, 100))
    dataset[0]