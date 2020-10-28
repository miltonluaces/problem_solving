import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    import h5py
import tensorflow as tf
from subprocess import check_output
    

def run(var):
    sess = tf.InteractiveSession()
    print("\n", var.name[:-2], ":\n",  var.eval())

def runStr(var, str):
    sess = tf.InteractiveSession()
    print("\n", str, ":\n",  sess.run(var))

def Run(list):
    sess = tf.InteractiveSession()
    sess.as_default()
    for var in list:
        print("\n", var.name[:-2], ":\n",  var.eval())

def Tboard(): 
    writer = tf.summary.FileWriter('.')
    writer.add_graph(tf.get_default_graph())
    check_output("tensorboard --logdir=C:/Users/m.martinez.luaces/source/repos/Tfbooks/TfCookbook/ --port 6006", shell=True)
    