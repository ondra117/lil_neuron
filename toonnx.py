import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tf2onnx
from model import Unet


onnx_model_name = 'model.onnx'

model = Unet(n_filters=[32, 64, 128, 256])
model.summary()
tf2onnx.convert.from_keras(model, output_path=onnx_model_name)