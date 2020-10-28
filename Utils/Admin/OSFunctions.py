#nvcc --version

import os
import tensorflow as tf
from tensorflow.python.client import device_lib


# Current working directory
cwd = os.getcwd()
print(cwd)

#os.environ['CUDA_VISIBLE_DEVICES'] = 1

# Check GPU
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
output = sess.run(hello)
print(output)

with tf.device('/cpu:0'):
    hello2 = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    output = sess.run(hello2)
    print(output)



def GetAvailableGPUs():
    devices = device_lib.list_local_devices()
    print(devices)
    return [x.name for x in devices if x.device_type == 'GPU']

gpus = GetAvailableGPUs()
print(gpus)

print("GPU available? " , tf.test.is_gpu_available())
print("GPU name : ", tf.test.gpu_device_name())