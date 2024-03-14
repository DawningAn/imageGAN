import os
import tensorflow as tf
from tensorflow.python.client import device_lib

tf.test.is_gpu_available()
print(tf.__version__)  # 查看TensorFlow的版本
print(tf.test.is_built_with_cuda()) # 判断CUDA是否可用
print(tf.test.is_gpu_available())  # 查看cuda、TensorFlow_GPU和cudnn(选择下载，cuda对深度学习的补充)版本是否对应

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)