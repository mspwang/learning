import tensorflow as tf
from tensorflow.python.platform import gfile
from google.protobuf import text_format

def load_graph_from_pbtxtfile(filename):
  with open(filename, 'r') as f:
    graph_def = tf.GraphDef()
    file_content = f.read()
    text_format.Merge(file_content, graph_def)
    #tf.import_graph_def(graph_def, name='')
    #tf.train.write_graph(graph_def, 'pbtxt/', filename + '.pb', as_text=False)
  return graph_def

def load_graph_from_pbfile(filename): 
  with gfile.FastGFile(filename,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    #tf.import_graph_def(graph_def, name='')
    #tf.train.write_graph(graph_def, 'pbtxt/', filename + '.pbtxt', as_text=True)
  return graph_def
