from keras import Model
from keras.layers import Layer, Input, Dense, Activation, Reshape, Conv1D, GlobalAveragePooling1D, LayerNormalization, Concatenate, BatchNormalization, Dot, UpSampling1D, Reshape, Flatten
import tensorflow as tf

class SinePositionEncoding(Layer):
    def __init__(self, n_dims, max_wavelength=10000, **kwargs,):
        super().__init__(**kwargs)
        self.n_dims = n_dims
        self.max_wavelength = max_wavelength
    
    def call(self, inputs):
        position = tf.cast(tf.reshape(inputs, (-1,)), self.compute_dtype)
        min_freq = tf.cast(1 / self.max_wavelength, self.compute_dtype)
        timescales = tf.pow(
            min_freq,
            tf.cast(2 * (tf.range(self.n_dims) // 2), self.compute_dtype)
            / tf.cast(self.n_dims, self.compute_dtype),
        )
        angles = tf.expand_dims(position, 1) * tf.expand_dims(timescales, 0)
        
        cos_mask = tf.cast(tf.range(self.n_dims) % 2, self.compute_dtype)
        sin_mask = 1 - cos_mask
        
        positional_encodings = (
            tf.sin(angles) * sin_mask + tf.cos(angles) * cos_mask
        )
        
        return positional_encodings
    
class CrossAttention(Layer):
    def __init__(self, heads=8, **kwargs):
        self.heads = heads
        super(CrossAttention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.convq = Dense(self.heads)
        self.convk = Dense(self.heads)
        self.convv = Dense(self.heads)
        self.convout = Dense(input_shape[0][-1])
        
        super(CrossAttention, self).build(input_shape)
        
    def call(self, inputs):
        x, context = inputs

        q = self.convq(x)
        k = self.convk(context)
        v = self.convv(context)
        
        scores = Dot(axes=-1)([q, k])
        
        weights = Activation("softmax")(scores)
        
        out = Dot(axes=(-1, 1))([weights, v])
        
        out = self.convout(out)
        
        return out

def resnet_block(x, n_filters, t=None, c=None):
    # block1
    x = BatchNormalization()(x)
    x = Activation("silu")(x)
    x = Conv1D(n_filters, 3, padding="same")(x)
    if not c is None:
        x += CrossAttention(heads=n_filters * 2)([x, c])
    x = BatchNormalization()(x)
    if not t is None:
        t_init_block = Activation("silu")(t)
        t_init_block = Dense(n_filters * 2)(t_init_block)
        t_init_block = t_init_block[:, None]
        shift = t_init_block[..., :n_filters]
        scale = t_init_block[..., n_filters:]
        scale += 1
        shift = tf.tile(shift, (1, x.shape[1], 1))
        scale = tf.tile(scale, (1, x.shape[1], 1))
        x *= scale
        x += shift
    x = Activation("silu")(x)
    x = Conv1D(n_filters, 3, padding="same")(x)
    return x

class Unet(Model):
    def __init__(self, input_size=(512, 1), n_filters=[32, 64, 128, 256, 512, 1024], n_resnet_blocks=2, word_dims=100, max_words=5, strides=2):

        inp_sound = Input(input_size)
        time = Input(shape=(1,))
        sound_time = Input(shape=(1,))
        generated_sound = Input(shape=(None, 1))
        text_embed = Input(shape=(max_words, word_dims))

        time_hiddens = SinePositionEncoding(word_dims)(time)
        time_hiddens = Dense(word_dims * 4)(time_hiddens)
        time_hiddens = Activation("silu")(time_hiddens)

        t = Dense(word_dims * 4)(time_hiddens)

        time_tokens = Dense(word_dims * max_words)(time_hiddens)
        time_tokens = Reshape((max_words, word_dims))(time_tokens)

        sound_time_hiddens = SinePositionEncoding(word_dims)(sound_time)
        sound_time_hiddens = Dense(word_dims * 4)(sound_time_hiddens)
        sound_time_hiddens = Activation("silu")(sound_time_hiddens)

        st = Dense(word_dims * 4)(sound_time_hiddens)
        t += st

        sound_time_tokens = Dense(word_dims * max_words)(sound_time_hiddens)
        sound_time_tokens = Reshape((max_words, word_dims))(sound_time_tokens)

        text_tokens = Flatten()(text_embed)
        text_tokens = Dense(word_dims * max_words, use_bias=False)(text_tokens)
        text_tokens = Reshape((max_words, word_dims))(text_tokens)

        text_hidens = GlobalAveragePooling1D()(text_tokens)
        text_hidens = LayerNormalization()(text_hidens)
        text_hidens = Dense(word_dims * 4)(text_hidens)
        text_hidens = Activation("silu")(text_hidens)
        text_hidens = Dense(word_dims * 4, use_bias=False)(text_hidens)
        t += text_hidens

        gst = Conv1D(n_filters[0], 3, padding="same")(generated_sound)
        gst = resnet_block(gst, n_filters=n_filters[0], c=sound_time_tokens)

        for filters in n_filters:
            gst = resnet_block(gst, n_filters=filters, c=sound_time_tokens)
            fft = tf.signal.fft(tf.cast(gst, dtype=tf.complex64))
            fft_real = tf.math.real(fft)
            fft_imag = tf.math.imag(fft)
            fft_tokens = Concatenate()([fft_real, fft_imag])
            fft_tokens = Dense(filters * 2)(fft_tokens)
            fft_tokens = Activation("swish")(fft_tokens)
            gst += CrossAttention(filters * 2)([gst, fft_tokens])
            for _ in range(n_resnet_blocks):
                gst = resnet_block(gst, n_filters=filters)
            conv = Conv1D(filters, strides, strides=strides)(gst)
            half = gst[:, -tf.shape(conv)[1] // strides:]
            gst = Concatenate()([conv, half])

        gst = resnet_block(gst, n_filters=filters, c=sound_time_tokens)
        for _ in range(n_resnet_blocks):
            gst = resnet_block(gst, n_filters=filters)

        gst = resnet_block(gst, n_filters=filters, c=sound_time_tokens)
        gst = resnet_block(gst, n_filters=filters, c=sound_time_tokens)

        gst_tokens = GlobalAveragePooling1D()(gst)
        gst_tokens = Flatten()(gst_tokens)
        gst_tokens = Dense(word_dims * max_words, use_bias=False)(gst_tokens)
        gst_tokens = Reshape((max_words, word_dims))(gst_tokens)

        c = Concatenate(axis=1)([text_tokens, time_tokens, sound_time_tokens, gst_tokens])
        c = LayerNormalization()(c)

        x = Conv1D(n_filters[0], 3, padding="same")(inp_sound)

        x = resnet_block(x, n_filters[0], t, c)

        hidens = []
        for filters in n_filters:
            x = resnet_block(x, filters, t, c)
            fft = tf.signal.fft(tf.cast(x, dtype=tf.complex64))
            fft_real = tf.math.real(fft)
            fft_imag = tf.math.imag(fft)
            fft_tokens = Concatenate()([fft_real, fft_imag])
            fft_tokens = Dense(filters * 2)(fft_tokens)
            fft_tokens = Activation("swish")(fft_tokens)
            x += CrossAttention(filters * 2)([x, fft_tokens])
            for _ in range(n_resnet_blocks):
                x = resnet_block(x, filters, t)
                hidens.append(x)
            x = x[:, ::strides, :]

        x = resnet_block(x, filters, t, c)
        for _ in range(n_resnet_blocks):
            x = resnet_block(x, filters, t)
            hidens.append(x)

        x = resnet_block(x, filters, t, c)
        x = resnet_block(x, filters, t, c)

        for filters in reversed(n_filters):
            x = resnet_block(x, filters, t, c)
            fft = tf.signal.fft(tf.cast(x, dtype=tf.complex64))
            fft_real = tf.math.real(fft)
            fft_imag = tf.math.imag(fft)
            fft_tokens = Concatenate()([fft_real, fft_imag])
            fft_tokens = Dense(filters * 2)(fft_tokens)
            fft_tokens = Activation("swish")(fft_tokens)
            x += CrossAttention(filters * 2)([x, fft_tokens])
            for _ in range(n_resnet_blocks):
                skip = hidens.pop()
                x = Concatenate()([x, skip])
                x = resnet_block(x, filters, t)
            x = UpSampling1D(size=strides)(x)

        x = resnet_block(x, filters, t, c)
        for _ in range(n_resnet_blocks):
            skip = hidens.pop()
            x = Concatenate()([x, skip])
            x = resnet_block(x, filters, t)

        x = Conv1D(input_size[-1], 3, padding="same")(x)

        super().__init__([inp_sound, time, sound_time, generated_sound, text_embed], x)

if __name__ == "__main__":
    import os
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # model = Unet(n_filters=[32, 64, 128])
    model = Unet()
    model.summary()