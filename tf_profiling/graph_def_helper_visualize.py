import graph_def_helper as helper

h = helper.GraphDefHelper()
import tensorflow as tf

_=tf.contrib
_=tf.core

from tensorflow.python.grappler import tf_optimizer

h.visualize("graph_after_rewrite2", None, None) 
