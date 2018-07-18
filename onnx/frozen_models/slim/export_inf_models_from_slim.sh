#/bin/bash

cur_path=`pwd`

source shared_env

model_name="$1"
checkpoint_host="$2"
checkpoint_link="$3"
label_offset="$4" # need to be enabled for vgg and resnetv1

echo current working directory is `pwd`
cpk_file=$checkpoint_host$checkpoint_link
echo downloading checkpoint file from $cpk_file
wget $cpk_file
echo extract checkpoint file, and delete the tar.gz file
mkdir $ckt_path/$model_name
tar -xvf $checkpoint_link -C $ckt_path/$model_name
rm $checkpoint_link
ckpt_ext_file_cnt=`ls -1 $ckt_path/$model_name/*.ckpt 2>/dev/null | wc -l`
if [ $ckpt_ext_file_cnt != 0 ]; then
  echo "*.ckpt file found for checkpoint"
  mv $ckt_path/$model_name/*.ckpt  $ckt_path/$model_name/model.ckpt 
else
  echo "the checkpoint file is not post-fixed with ckpt."
fi

input_checkpoint=$ckt_path/$model_name/model.ckpt 

echo exporting inference graph 
inference_graph_name=$model_name$model_postfix
echo change working directory to $model_git_path
output_file_path=$exported_inf_path/$inference_graph_name
echo export inference graph to $output_file_path
cd $model_git_path
python3 export_inference_graph.py  --alsologtostderr --model_name=$model_name --output_file=$output_file_path --labels_offset=$label_offset


echo change working directory to $tf_git_path
cd $tf_git_path
echo get the outputs name with summarize tool
bazel build tensorflow/tools/graph_transforms:summarize_graph

summarize_cmd="bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=$exported_inf_path/$inference_graph_name "
echo $summarize_cmd
outputs=`$summarize_cmd 2> /dev/null | grep -E "\-\-output_layer=" | sed -r 's/^.*--output_layer=(.*$)/\1/g'`
echo output_node_name is $outputs

echo freeze the graph based on downloaded checkpoint file
bazel build tensorflow/python/tools:freeze_graph
freeze_cmd="bazel-bin/tensorflow/python/tools/freeze_graph --input_graph=$output_file_path --input_checkpoint=$input_checkpoint --input_binary=true --output_graph=$frozen_inf_path/$inference_graph_name --output_node_names=$outputs"
echo $freeze_cmd 

result=`$freeze_cmd`


echo "all frozen fraph is stored at " $frozen_inf_path

echo switch back to original directory: $cur_path
cd $cur_path
