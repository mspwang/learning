import tensorflow as tf
import json
from tensorflow.python.client import timeline
from tensorflow.python.platform import gfile
from google.protobuf import text_format
from graph_def_loader import load_graph_from_pbtxtfile, load_graph_from_pbfile

class TimeLiner:
    _timeline_dict = None

    def update_timeline(self, chrome_trace):
        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)

class MemProfiling:
    def __init__(self):
        pass

    def profile(self, graph_def_filename, chrome_trace_filename, target_node_name, feed_dict=None):
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            graph_def = load_graph_from_pbtxtfile('./' + graph_def_filename + '.pbtxt')
            g_in = tf.import_graph_def(graph_def, name='')

        train_writer = tf.summary.FileWriter('./')
        train_writer.add_graph(sess.graph)

        train_writer.flush()
        train_writer.close()

        #sess.run(tf.global_variables_initializer())
        #target = sess.graph.get_tensor_by_name('e:0')
        target = sess.graph.get_operation_by_name(target_node_name)#'d_copy_to_cpu_dep')
        init_op = sess.graph.get_operation_by_name('init')

        feeds = {}
        for feed_tensor in feed_dict:
            tensor = sess.graph.get_tensor_by_name(feed_tensor.name)
            if feed_tensor is None:
                raise Exception("can not found the tensor for feed")
            feeds[tensor] = feed_dict[feed_tensor]
            print(tensor)
            print(feed_tensor)
        run_metadata = tf.RunMetadata()
        sess.run(init_op, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata, feed_dict=feeds)
        many_runs_timeline = TimeLiner()

        runs = 1
        for i in range(runs):
            sess.run(target, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata, feed_dict=feeds)
            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            chrome_trace = trace.generate_chrome_trace_format()
            many_runs_timeline.update_timeline(chrome_trace)

        many_runs_timeline.save(chrome_trace_filename + '.ctf.json')
