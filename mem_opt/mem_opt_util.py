import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes
import queue
 
class MemOptUtil:
    def __init__(self, graph_def, target_node_name):
        # todo: make this automatically
        assert(target_node_name != None)
        self.target_node = None # the node, that is the final output before the gradients are added.
        self.graph_def = graph_def
        self.node_map_by_name = {}

        self.node_outputs = {} # name=>reference count , only for those intermediate node
        for node in self.graph_def.node:
            self.node_map_by_name[node.name] = node

            if node.name not in self.node_outputs:
                self.node_outputs[node.name] = []
            if node.input:
                for input_node_name in node.input:
                    if input_node_name not in self.node_outputs:
                        self.node_outputs[input_node_name] = []
                    self.node_outputs[input_node_name].extend([node.name])
        self.target_node = self.node_map_by_name[target_node_name]
        self.intermediate_nodes = self.find_intermediate_nodes(self.target_node)

    def optimize_graph(self, filename):
        graph_def_temp = self.graph_def
        modified_graph_def = self.modify_graph(graph_def_temp)
        tf.train.write_graph(modified_graph_def, '.', filename + '.pbtxt', as_text=True)
        tf.train.write_graph(modified_graph_def, '.', filename + '.pb', as_text=False)

    def add_swap_operators(self, graph_def, intermediate_node, node_name_to_insert_after):
        print("intermediate_node : ")
        print(intermediate_node)
        if node_name_to_insert_after == self.target_node.name:
            print("skip add control-depencecy and copy-to-cpu node on target node, because we assume this data will be used very soon")
            return
        
        dtype = dtypes.float32.as_datatype_enum
        if intermediate_node.op == 'Equal':
            dtype = dtypes.bool.as_datatype_enum
        elif intermediate_node.op == 'Shape':
            dtype = intermediate_node.attr['out_type'].type
        

        swap_out_node = node_def_pb2.NodeDef()
        swap_out_node.op = 'Identity'
        swap_out_node.name = intermediate_node.name + "_on_cpu"
        # Todo: if we can explicitly release the intermediate after copy, we might make sure swap_out_node 
        # dpends on intermediate data finished it's mission (e.g. control dependency on last calculate it is involved)
        swap_out_node.input.extend([intermediate_node.name])
        swap_out_node.device = "/device:CPU:0"
        swap_out_node.attr['T'].CopyFrom(
            attr_value_pb2.AttrValue(type=dtype))    


        ctrl_node = node_def_pb2.NodeDef()
        ctrl_node.op = 'NoOp'
        ctrl_node.name = intermediate_node.name + '_copy_to_cpu_dep'
        ctrl_node.device = "/device:CPU:0"
        ctrl_node.input.extend(['^' + swap_out_node.name])
        
        for node_name in self.node_outputs[node_name_to_insert_after]:
            # ignore those gradients operators when we try to add control-dependency node
            if not self.is_gradient_operator(node_name):
                node = self.node_map_by_name[node_name]
                node.input.extend(['^' + ctrl_node.name]) 

        swap_in_node = node_def_pb2.NodeDef()
        swap_in_node.op = 'Identity'
        swap_in_node.name = intermediate_node.name + '_copied_back_gpu'
        swap_in_node.input.extend([swap_out_node.name])
        swap_in_node.attr['T'].CopyFrom(
            attr_value_pb2.AttrValue(type=dtype))    

        for node in graph_def.node:
            if node.name.startswith('gradients') or node.name.startswith('GradientDescent'):
                input_names = node.input
                new_input_names = []
                for input_node_name in input_names:
                    if input_node_name == intermediate_node.name:
                        new_input_names.append(swap_in_node.name)
                    elif input_node_name == '^' + intermediate_node.name:
                        new_input_names.append('^' + swap_in_node.name)
                    else:
                        new_input_names.append(input_node_name)
                del node.input[:]
                node.input.extend(new_input_names)
            else:
                continue

        graph_def.node.extend([swap_out_node, ctrl_node, swap_in_node])
        return graph_def


    def find_intermediate_nodes(self, target_node):
        non_intermediate_node_type = ['Const', 'VariableV2', 'Identity', 'Assign', 'Placeholder']
        q = queue.Queue()
        q.put([target_node, None])

        already_scanned = [target_node.name]
        iteration_count = 0
        intermediate_nodes = {}
        while not q.empty():
            front, node_to_output = q.get()
            
            parent_name = "" 
            if node_to_output:
                parent_name = node_to_output.name
            print(front.name + ", output to node : " + parent_name)
            if front.op in non_intermediate_node_type:
                pass
                print(front.name + ':[' + front.op +'] Not intermediate node type, ignore it and its inputs...')
                continue
            
            # ignore the last node, because that's useless for calculating gradients
            if iteration_count > 0:
                # parent name means: after the node, we can copy current node's result.
                if front.name not in intermediate_nodes:
                    intermediate_nodes[front.name] = parent_name
                    print(front.name + ':[' + front.op +'] is intermediate node...')
                else:
                    assert(1 == 0)
            
            iteration_count += 1
            if front.input:
                for ip in front.input:
                    if ip.startswith("^"):
                        print("We assume conditional ops will not return tensor as result, so just ignore the search along this path.")
                        continue
                    if ip not in already_scanned:
                        n = self.node_map_by_name[ip]
                        assert(n != None)
                        already_scanned.extend([ip])
                        q.put([n, self.node_map_by_name[front.name]])
                    else:
                        print(ip + ' already scanned by previous step, skip...')

        return intermediate_nodes

    def modify_graph(self, graph_def):
        # from target node, check its inputs: if the input node is 1). NOT Const and 2). NOT Initial Variables, then we will mark them as intermediate nodes to swap in/out
        target_node = None # to find the node of e
        ''' note: we CANNOT use train, since it is the target node that count in gradient related ops added by optimizer'''

        print("start modify_graph...")
        nodes_snapshot = graph_def.node
        for node in nodes_snapshot:
            if node.name in self.intermediate_nodes:
                print(node.name+' is beding processed...')
                node_name_to_insert_after = self.intermediate_nodes[node.name]
                self.add_swap_operators(graph_def, node, node_name_to_insert_after)
        return graph_def

    def is_gradient_operator(self, ops_name):
        if ops_name.startswith('gradients') or ops_name.startswith('GradientDescent'):
            return True
        return False
