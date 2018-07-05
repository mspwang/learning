import tensorflow as tf
from tensorflow.python.platform import gfile
from google.protobuf import text_format

def convert_pbtxt_to_pb(filename, dest_name):
  with open(filename, 'r') as f:
    graph_def = tf.GraphDef()
    file_content = f.read()
    text_format.Merge(file_content, graph_def)
    tf.import_graph_def(graph_def, name='')
    tf.train.write_graph(graph_def, './', dest_name + '.pb', as_text=False)
  return graph_def

def convert_pb_to_pbtxt(filename, dest_name): 
  with gfile.FastGFile(filename,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
    tf.train.write_graph(graph_def, './', dest_name + '.pbtxt', as_text=True)
  return graph_def

import argparse
def run_main():
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--src",
        type=str,
        default=""
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=""
    )
    parser.add_argument(
        "--to_txt",
        type="bool",
        default=True
    )
    flags, unparsed = parser.parse_known_args()
    print(flags)

    if (not flags.src or not flags.dest):
        print("no source file or destination file specified, exit.")
        return

    if flags.to_txt == True:
        print("convert pb file to txt file")
        convert_pb_to_pbtxt(flags.src, flags.dest)
    else:
        print("convert pb txt file to pb file")
        convert_pbtxt_to_pb(flags.src, flags.dest)

if __name__ == "__main__":
    run_main()

