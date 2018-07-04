#/bin/bash

cur_path=`pwd`
pb_file=$1
source shared_env

echo change working directory to $tf_git_path
cd $tf_git_path
ls /tmp/frozen/ > /tmp/temp_frozen.txt
bazel build tensorflow/tools/graph_transforms:summarize_graph

while IFS='' read -r line || [[ -n "$line" ]]; do
  if [ ! -z $pb_file -a $pb_file != $line ]; then
    continue
  fi
  echo -------------------- start handling $line ----------------------------
  echo change working directory to $tf_git_path
  cd $tf_git_path

  echo ------ summarize the frozen graph, to get the inputs and outputs name
  summarize_cmd="bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=/tmp/frozen/$line"
  echo $summarize_cmd
  bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=/tmp/frozen/$line
  outputs=`$summarize_cmd 2> /dev/null | grep -E "\-\-output_layer=" | sed -r 's/^.*--output_layer=(.*$)/\1/g'`
  inputs=`$summarize_cmd 2> /dev/null | grep -E "\-\-input_layer=" | sed -r 's/^.*--input_layer=(.*)[[:space:]]--input_layer_type.*$/\1/g'`

  echo ------ update the inputs and outputs name to format like "input_name:index"
  update_output_name=`python3 $rename_script_path $outputs  2> /dev/null`
  update_input_name=`python3 $rename_script_path $inputs  2> /dev/null`
  echo python3 $rename_script_path $inputs  2> /dev/null
  echo updated input names is $update_input_name, output names is $update_output_name

  echo ------ start convertion, tensorflow usage require caller program must not in tensorflow root folder, so switch to current user directory with cd
  cd 
  python3 -m tf2onnx.convert --input "/tmp/frozen/$line" --inputs $update_input_name --outputs $update_output_name --output $line.onnx --verbose
  echo python3 -m tf2onnx.convert --input "/tmp/frozen/$line" --inputs $update_input_name --outputs $update_output_name --output $line.onnx --verbose
done < /tmp/temp_frozen.txt

echo ------ switch back to original directory: $cur_path
cd $cur_path
