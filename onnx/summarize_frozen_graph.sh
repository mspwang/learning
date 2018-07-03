#/bin/bash

tf_git_path="/home/pengwa/community/tensorflow"

echo change working directory to $tf_git_path
cd $tf_git_path
ls /tmp/frozen/ > /tmp/temp_frozen.txt
bazel build tensorflow/tools/graph_transforms:summarize_graph

while IFS='' read -r line || [[ -n "$line" ]]; do
  echo -------------------- start handling $line ----------------------------
  summarize_cmd="bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=/tmp/frozen/$line"
  echo $summarize_cmd
  bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=/tmp/frozen/$line
  outputs=`$summarize_cmd 2> /dev/null | grep -E "\-\-output_layer=" | sed -r 's/^.*--output_layer=(.*$)/\1/g'`
  inputs=`$summarize_cmd 2> /dev/null | grep -E "\-\-input_layer=" | sed -r 's/^.*--input_layer=(.*)[[:space:]]--input_layer_type.*$/\1/g'`
  update_output_name=`python3 /home/pengwa/community/update_output_name.py $outputs  2> /dev/null`
  update_input_name=`python3 /home/pengwa/community/update_output_name.py $inputs  2> /dev/null`
  echo updated input names is $update_input_name, output names is $update_output_name
  echo start convertion
  echo python3 -m tfonnx.convert --input "/tmp/frozen/$line" --inputs $update_input_name --outputs $update_output_name --output $line.onnx --verbose
  python3 -m tfonnx.convert --input "/tmp/frozen/$line" --inputs $update_input_name --outputs $update_output_name --output $line.onnx --verbose
done < /tmp/temp_frozen.txt
