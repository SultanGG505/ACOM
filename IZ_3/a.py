import tensorflow as tf

assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
