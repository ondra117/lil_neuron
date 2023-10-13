import numpy as np
import tensorflow as tf

class Sampler:
    def __init__(self):
        ...
        
    @classmethod
    def cosine(self, n_timesteps, s, tensor=False):
        self = self()
        
        self.n_timesteps = n_timesteps
        self.tensor = tensor
        
        t = np.arange(0, n_timesteps + 1)
        
        ft = t / n_timesteps + s
        ft /= 1 + s
        ft *= np.pi / 2
        ft = np.cos(ft)
        ft = np.power(ft, 2)
        
        cumulative_alphas = ft / (np.cos(s / (s + 1) * np.pi / 2) ** 2)
        
        self.betas = 1 - cumulative_alphas[1:] / cumulative_alphas[:1]
        
        self.betas = np.pad(self.betas, (1, 0), constant_values=0)
        
        self.alphas = 1 - self.betas
        
        self.sqer_alphas = np.sqrt(self.alphas)
        
        self.sqer_betas = np.sqrt(self.betas)
        
        self.sqer_cumulative_alphas = np.sqrt(cumulative_alphas)
        
        self.sqer_one_minus_cumulative_alphas = np.sqrt(1 - cumulative_alphas)

        if self.tensor:
            self.sqer_cumulative_alphas = tf.convert_to_tensor(self.sqer_cumulative_alphas, dtype=tf.float32)
            self.sqer_one_minus_cumulative_alphas = tf.convert_to_tensor(self.sqer_one_minus_cumulative_alphas, dtype=tf.float32)

        return self
    
    def q_sample(self, x0, t):
        if self.tensor:
            noise = tf.random.normal(tf.shape(x0), 0, 1)
            xt = x0 * tf.gather(self.sqer_cumulative_alphas, t) + noise * tf.gather(self.sqer_one_minus_cumulative_alphas, t)
        else:
            noise = np.random.normal(0, 1, size=x0.shape)
            xt = x0 * self.sqer_cumulative_alphas[t] + noise * self.sqer_one_minus_cumulative_alphas[t]
        return xt, noise
    
    def step_beck(self, xt, noise, t, add_noise=True):
        if self.tensor:
            values = tf.gather(self.sqer_one_minus_cumulative_alphas, t)
            x0 = noise * (values + 1e-8)
        else:
            x0 = noise * self.sqer_one_minus_cumulative_alphas[t]
        x0 = xt - x0
        if self.tensor:
            values = tf.gather(self.sqer_cumulative_alphas, t)
            x0 /= values
        else:
            x0 /= self.sqer_cumulative_alphas[t]
        
        if add_noise:
            out, _ = self.q_sample(x0, t - 1)
        else:
            out = x0
        
        return out