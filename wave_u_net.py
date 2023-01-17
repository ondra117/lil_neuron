# -*- coding: utf-8 -*-
"""Wave-U-net.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hiok2PcpQZl3R01kf-V6yaN-tKfkD4Z3
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer

"""# Custom layers"""

class SelfAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(SelfAttention, self).__init__(**kwargs)

  def build(self, input_shape):
    # Vrstva pro výpočet váh pozornosti
    self.dense = tf.keras.layers.Dense(1, activation='tanh')
    # Vrstva pro normalizaci váh pozornosti pomocí funkce softmax
    self.flatten = tf.keras.layers.Flatten()
    self.activation = tf.keras.layers.Activation('softmax')
    self.reshape = tf.keras.layers.Reshape((-1, 1))
    super(SelfAttention, self).build(input_shape)

  def call(self, inputs, training=None, mask=None):
    # Váhy pozornosti
    attention_weights = self.dense(inputs)
    attention_weights = self.flatten(attention_weights)
    attention_weights = self.activation(attention_weights)
    attention_weights = self.reshape(attention_weights)
    # Vážený vstup
    weighted_input = tf.keras.layers.multiply([inputs, attention_weights])
    return weighted_input

class PolarizedSelfAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(PolarizedSelfAttention, self).__init__(**kwargs)

  def build(self, input_shape):
    self.conv1 = tf.keras.layers.Conv1D(filters=1, kernel_size=1)
    self.conv2 = tf.keras.layers.Conv1D(filters=input_shape[-1] // 2, kernel_size=1)
    self.reshape1 = tf.keras.layers.Reshape((1, -1))
    self.softmax = tf.keras.layers.Activation("softmax")
    self.dot1 = tf.keras.layers.Dot(axes=(2, 1))
    self.conv3 = tf.keras.layers.Conv1D(filters=input_shape[-1], kernel_size=1)
    #norm
    self.sigmoid = tf.keras.layers.Activation("sigmoid")
    self.attention1 = tf.keras.layers.Attention(score_mode='dot')

    self.conv4 = tf.keras.layers.Conv1D(filters=input_shape[-1] // 2, kernel_size=1)
    self.conv5 = tf.keras.layers.Conv1D(filters=input_shape[-1] // 2, kernel_size=1)
    self.maxpool = tf.keras.layers.GlobalMaxPooling1D()
    self.reshape2 = tf.keras.layers.Reshape((-1, 1))
    self.dot2 = tf.keras.layers.Dot(axes=(1, 2))

    self.reshape3 = tf.keras.layers.Reshape((input_shape[-1], -1))
    self.attention2 = tf.keras.layers.Attention(score_mode='dot')
    self.reshape4 = tf.keras.layers.Reshape((-1, input_shape[-1]))
    super(PolarizedSelfAttention, self).build(input_shape)

  def call(self, inputs, training=None, mask=None):
    t1 = self.conv1(inputs)
    t2 = self.conv2(inputs)
    t1 = self.reshape1(t1)
    t1 = self.softmax(t1)

    t2 = self.dot1([t1, t2])
    t2 = self.conv3(t2)
    #norm
    t2 = self.sigmoid(t2)

    o = self.attention1([inputs, t2])


    t1 = self.conv4(o)
    t2 = self.conv5(o)
    t1 = self.maxpool(t1)
    t1 = self.reshape2(t1)

    t2 = self.dot2([t1, t2])#wrd nwm

    t2 = self.sigmoid(t2)

    o = self.reshape3(o)#nwm
    output = self.attention2([o, t2])

    output = self.reshape4(output)#nwm

    return output



class AudioClipLayer(Layer):

    def __init__(self, **kwargs):
        '''Initializes the instance attributes'''
        super(AudioClipLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        '''Create the state of the layer (weights)'''
        # initialize the weights
        pass
        
    def call(self, inputs, training):
        '''Defines the computation from inputs to outputs'''
        if training:
            return inputs
        else:
            return tf.maximum(tf.minimum(inputs, 1.0), -1.0)

# Learned Interpolation layer

class InterpolationLayer(Layer):

    def __init__(self, padding = "valid", **kwargs):
        '''Initializes the instance attributes'''
        super(InterpolationLayer, self).__init__(**kwargs)
        self.padding = padding

    def build(self, input_shape):
        '''Create the state of the layer (weights)'''
        self.features = input_shape.as_list()[3]

        # initialize the weights
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(name="kernel",
            initial_value=w_init(shape=(self.features, ),
                                 dtype='float32'),
            trainable=True)

    def call(self, inputs):
        '''Defines the computation from inputs to outputs'''

        w_scaled = tf.math.sigmoid(self.w)

        counter_w = 1 - w_scaled

        conv_weights = tf.expand_dims(tf.concat([tf.expand_dims(tf.linalg.diag(w_scaled), axis=0), tf.expand_dims(tf.linalg.diag(counter_w), axis=0)], axis=0), axis=0)

        intermediate_vals = tf.nn.conv2d(inputs, conv_weights, strides=[1,1,1,1], padding=self.padding.upper())

        intermediate_vals = tf.transpose(intermediate_vals, [2, 0, 1, 3])
        out = tf.transpose(inputs, [2, 0, 1, 3])
        
        num_entries = out.shape.as_list()[0]
        out = tf.concat([out, intermediate_vals], axis=0)

        indices = list()

        # num_outputs = 2*num_entries - 1
        num_outputs = (2*num_entries - 1) if self.padding == "valid" else 2*num_entries

        for idx in range(num_outputs):
            if idx % 2 == 0:
                indices.append(idx // 2)
            else:
                indices.append(num_entries + idx//2)
        out = tf.gather(out, indices)
        current_layer = tf.transpose(out, [1, 2, 0, 3])

        return current_layer

class CropLayer(Layer):
    def __init__(self, x2, match_feature_dim=True, **kwargs):
        '''Initializes the instance attributes'''
        super(CropLayer, self).__init__(**kwargs)
        self.match_feature_dim = match_feature_dim
        self.x2 = x2

    def build(self, input_shape):
        '''Create the state of the layer (weights)'''
        # initialize the weights
        pass
        
    def call(self, inputs):
        '''Defines the computation from inputs to outputs'''
        if self.x2 is None:
            return inputs

        inputs = self.crop(inputs, self.x2.shape.as_list(), self.match_feature_dim)
        return inputs

    def crop(self, tensor, target_shape, match_feature_dim=True):
        '''
        Crops a 3D tensor [batch_size, width, channels] along the width axes to a target shape.
        Performs a centre crop. If the dimension difference is uneven, crop last dimensions first.
        :param tensor: 4D tensor [batch_size, width, height, channels] that should be cropped. 
        :param target_shape: Target shape (4D tensor) that the tensor should be cropped to
        :return: Cropped tensor
        '''
        shape = np.array(tensor.shape.as_list())

        ddif = shape[1] - target_shape[1]

        if (ddif % 2 != 0):
            print("WARNING: Cropping with uneven number of extra entries on one side")
        # assert diff[1] >= 0 # Only positive difference allowed
        if ddif == 0:
            return tensor
        crop_start = ddif // 2
        crop_end = ddif - crop_start

        return tensor[:,crop_start:-crop_end,:]

class IndependentOutputLayer(Layer):

    def __init__(self, source_names, num_channels, filter_width, padding="valid", **kwargs):
        '''Initializes the instance attributes'''
        super(IndependentOutputLayer, self).__init__(**kwargs)
        self.source_names = source_names
        self.num_channels = num_channels
        self.filter_width = filter_width
        self.padding = padding

        self.conv1a = tf.keras.layers.Conv1D(self.num_channels, self.filter_width, padding= self.padding)


    def build(self, input_shape):
        '''Create the state of the layer (weights)'''
        pass
        
    def call(self, inputs, training):
        '''Defines the computation from inputs to outputs'''
        outputs = {}
        for name in self.source_names:
            out = self.conv1a(inputs)
            outputs[name] = out
        
        return outputs

class DiffOutputLayer(Layer):

    def __init__(self, source_names, num_channels, filter_width, padding="valid", **kwargs):
        '''Initializes the instance attributes'''
        super(DiffOutputLayer, self).__init__(**kwargs)
        self.source_names = source_names
        self.num_channels = num_channels
        self.filter_width = filter_width
        self.padding = padding

        self.conv1a = tf.keras.layers.Conv1D(self.num_channels, self.filter_width, padding= self.padding)


    def build(self, input_shape):
        '''Create the state of the layer (weights)'''
        pass
        
    def call(self, inputs, training):
        '''Defines the computation from inputs to outputs'''
        outputs = {}
        sum_source = 0
        for name in self.source_names[:-1]:
            out = self.conv1a(inputs[0])
            out = AudioClipLayer()(out)
            outputs[name] = out
            sum_source = sum_source + out
        
        last_source = CropLayer(sum_source)(inputs[1]) - sum_source
        last_source = AudioClipLayer()(last_source)

        outputs[self.source_names[-1]] = last_source

        return outputs

"""# Define the Network"""

def wave_u_net(num_initial_filters = 24, num_layers = 12, kernel_size = 15, merge_filter_size = 5, 
               source_names = ["bass", "drums", "other", "vocals"], num_channels = 1, output_filter_size = 1,
               padding = "same", input_size = 16384 * 4, context = False, upsampling_type = "learned",
               output_activation = "linear", output_type = "difference", attention = "False", dropout = False, dropout_rate = 0.2):
  
  # `enc_outputs` stores the downsampled outputs to re-use during upsampling.
  enc_outputs = []

  # `raw_input` is the input to the network
  raw_input = tf.keras.layers.Input(shape=(input_size, num_channels),name="raw_input")
  X = raw_input
  inp = raw_input

  # Down sampling
  for i in range(num_layers):
    X = tf.keras.layers.Conv1D(filters=num_initial_filters + (num_initial_filters * i),
                          kernel_size=kernel_size,strides=1,
                          padding=padding, name="Down_Conv_"+str(i))(X)
    X = tf.keras.layers.LeakyReLU(name="Down_Conv_Activ_"+str(i))(X)

    if dropout:
      X = tf.keras.layers.Dropout(rate=dropout_rate, name="Down_Dropout_"+str(i))(X)

    enc_outputs.append(X)

    X = tf.keras.layers.Lambda(lambda x: x[:,::2,:], name="Decimate_"+str(i))(X)


  X = tf.keras.layers.Conv1D(filters=num_initial_filters + (num_initial_filters * num_layers),
                          kernel_size=kernel_size,strides=1,
                          padding=padding, name="Down_Conv_"+str(num_layers))(X)
  X = tf.keras.layers.LeakyReLU(name="Down_Conv_Activ_"+str(num_layers))(X)

  if dropout:
    X = tf.keras.layers.Dropout(rate=dropout_rate, name="Down_Dropout_"+str(num_layers))(X)



  # Up sampling
  for i in range(num_layers):
    X = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1), name="exp_dims_"+str(i))(X)
    
    if upsampling_type == "learned":
      X = InterpolationLayer(name="IntPol_"+str(i), padding=padding)(X)

    else:
      if context:
        X = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, [1, x.shape.as_list()[2] * 2 - 1]), name="bilinear_interpol_"+str(i))(X)
        # current_layer = tf.image.resize_bilinear(current_layer, [1, current_layer.get_shape().as_list()[2] * 2 - 1], align_corners=True)
      else:
        X = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, [1, x.shape.as_list()[2] * 2]), name="bilinear_interpol_"+str(i))(X)
        # current_layer = tf.image.resize_bilinear(current_layer, [1, current_layer.get_shape().as_list()[2]*2]) # out = in + in - 1


    X = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1), name="sq_dims_"+str(i))(X)
    
    c_layer = CropLayer(X, False, name="crop_layer_"+str(i))(enc_outputs[-i-1])

    if attention == "Normal":
        c_layer = SelfAttention(name="Attention_"+str(i))(c_layer)
    elif attention == "Polarized":
        c_layer = PolarizedSelfAttention(name="Pol_Attention_"+str(i))(c_layer)

    X = tf.keras.layers.Concatenate(axis=2, name="concatenate_"+str(i))([X, c_layer]) 


    X = tf.keras.layers.Conv1D(filters=num_initial_filters + (num_initial_filters * (num_layers - i - 1)),
                            kernel_size=merge_filter_size,strides=1,
                            padding=padding, name="Up_Conv_"+str(i))(X)
    X = tf.keras.layers.LeakyReLU(name="Up_Conv_Activ_"+str(i))(X)

    if dropout:
      X = tf.keras.layers.Dropout(rate=dropout_rate, name="Up_Dropout_"+str(i))(X)


  c_layer = CropLayer(X, False, name="crop_layer_"+str(num_layers))(inp)
  X = tf.keras.layers.Concatenate(axis=2, name="concatenate_"+str(num_layers))([X, c_layer]) 
  X = AudioClipLayer(name="audio_clip_"+str(0))(X)

  if output_type == "direct":
    X = IndependentOutputLayer(source_names, num_channels, output_filter_size, padding=padding, name="independent_out")(X)
  
  elif output_type == "single":
    X = tf.keras.layers.Conv1D(num_channels, output_filter_size, padding= padding, name="single_out")(X)
  else:
    # Difference Output
    cropped_input = CropLayer(X, False, name="crop_layer_"+str(num_layers+1))(inp)
    X = DiffOutputLayer(source_names, num_channels, output_filter_size, padding=padding, name="diff_out")([X, cropped_input])

  o = X
  model = tf.keras.Model(inputs=raw_input, outputs=o)
  return model

# Parameters for the Wave-U-net

params = {
  "num_initial_filters": 24,
  "num_layers": 12,
  "kernel_size": 15,
  "merge_filter_size": 5,
  "source_names": ["bass", "drums", "other", "vocals"],
  "num_channels": 2,
  "output_filter_size": 1,
  "padding": "valid",
  "input_size": 147443,
  "context": True,
  "upsampling_type": "learned",         # "learned" or "linear"
  "output_activation": "linear",        # "linear" or "tanh"
  "output_type": "difference",          # "direct" or "single" or "difference" 
  "attention": "False",                 # "False" or "Normal" or "Polarized"
  "dropout": False,
  "dropout_rate": 0.2
}


"""# Other utility functions"""

def get_padding(shape, num_layers=12, filter_size=15, input_filter_size=15, output_filter_size=1, merge_filter_size=5, num_channels=1, context = True):
    '''
    Note that this function is not used within the Wave-U-net. 
    But it is useful to calculate the required amounts of padding along 
    each axis of the input and output, so that the Unet works and has the 
    given shape as output shape.

    :param shape: Desired output shape 
    :return: Input_shape, output_shape, where each is a list [batch_size, time_steps, channels]
    '''

    if context:
        # Check if desired shape is possible as output shape - go from output shape towards lowest-res feature map
        rem = float(shape[1]) # Cut off batch size number and channel

        # Output filter size
        rem = rem - output_filter_size + 1

        # Upsampling blocks
        for i in range(num_layers):
            rem = rem + merge_filter_size - 1
            rem = (rem + 1.) / 2.# out = in + in - 1 <=> in = (out+1)/

        # Round resulting feature map dimensions up to nearest integer
        x = np.asarray(np.ceil(rem),dtype=np.int64)
        assert(x >= 2)

        # Compute input and output shapes based on lowest-res feature map
        output_shape = x
        input_shape = x

        # Extra conv
        input_shape = input_shape + filter_size - 1

        # Go from centre feature map through up- and downsampling blocks
        for i in range(num_layers):
            output_shape = 2*output_shape - 1 #Upsampling
            output_shape = output_shape - merge_filter_size + 1 # Conv

            input_shape = 2*input_shape - 1 # Decimation
            if i < num_layers - 1:
                input_shape = input_shape + filter_size - 1 # Conv
            else:
                input_shape = input_shape + input_filter_size - 1

        # Output filters
        output_shape = output_shape - output_filter_size + 1

        input_shape = np.concatenate([[shape[0]], [input_shape], [num_channels]])
        output_shape = np.concatenate([[shape[0]], [output_shape], [num_channels]])

        return input_shape, output_shape
    else:
        return [shape[0], shape[1], num_channels], [shape[0], shape[1], num_channels]