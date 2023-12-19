import tensorflow as tf
import os
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "2"
assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
