# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = None

def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784], name='x')
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.add(tf.matmul(x, W), b, name='y')

  # Define loss and optimizer
  y_ = tf.placeholder(tf.int64, [None], name='y_')

  cross_entropy_node_name = y.name.split(":")[0]
  #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  
  sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
  sess.run(tf.global_variables_initializer())

  from mem_opt_util import MemOptUtil
  from mem_profiling import MemProfiling
  from mem_profiling import TimeLiner
 
  file_writer = tf.summary.FileWriter('./org/', sess.graph)
  graph_def = sess.graph.as_graph_def(add_shapes=True)

  saved_graph_def_filename = "modified_mnist_softmax"
  tf.train.write_graph(graph_def, '.', 'un' + saved_graph_def_filename, as_text=True)

  opt_util = MemOptUtil(graph_def, cross_entropy_node_name)
  #print(sess.run(sess.graph.get_tensor_by_name('sparse_softmax_cross_entropy_loss/Equal:0')))
  opt_util.optimize_graph(saved_graph_def_filename)

  tf.reset_default_graph()
  mem_profile = MemProfiling()
  batch_xs, batch_ys = mnist.train.next_batch(100)
  mem_profile.profile(saved_graph_def_filename, saved_graph_def_filename + "_profiling", cross_entropy_node_name, feed_dict={x: batch_xs, y_: batch_ys})

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/tensorflow/mnist/input_data',
      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
