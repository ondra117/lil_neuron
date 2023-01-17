import tensorflow as tf
from keras.losses import Loss
import numpy as np

def combined_loss(y_true, y_pred):
    # Vypočtěte KLDivergence
    kd_loss = tf.keras.losses.KLDivergence()(y_true, y_pred)
    # Vypočtěte MeanSquaredError
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
  
    # Nalezněte maximum z KLDivergence
    max_kd_loss = tf.math.reduce_max(kd_loss)
    # Nalezněte maximum z MeanSquaredError
    max_mse_loss = tf.math.reduce_max(mse_loss)
  
    # Vypočtěte průměrné maximum z obou ztrát
    avg_max = (max_kd_loss + max_mse_loss) / 2
  
    # Normalizujte obě ztráty
    normalized_kd_loss = kd_loss / max_kd_loss
    normalized_mse_loss = mse_loss / max_mse_loss
  
    # Spojte obě ztráty a vynásobte je průměrným maximem
    total_loss = (normalized_kd_loss + normalized_mse_loss) * avg_max
    
    return total_loss

class ScatterLoss(Loss):
    def __init__(self, s_size, steps, max_scatter):
        super().__init__(name="scatter_loss")
        movs = s_size // steps
        interval = np.zeros([s_size], dtype=np.float32)

        for d in range(1, steps):
            interval[d * movs:] = d

        interval /= (steps - 1)
        interval *= max_scatter * (steps - 2) / (steps - 1)

        self.interval = tf.convert_to_tensor(interval.reshape([1, -1, 1]), dtype=tf.float32)

    def __call__(self, y_true, y_pred, sample_weight):
        xd = y_pred - y_true - self.interval
        p_side = tf.math.maximum(0.0, xd) / (xd)
        p_side *= tf.math.pow(xd, 2)

        dx = y_true - y_pred - self.interval
        l_side = tf.math.maximum(0.0, dx) / (dx)
        l_side *= tf.math.pow(dx, 2)

        loss = tf.reduce_sum(p_side + l_side, 0)
        return loss
