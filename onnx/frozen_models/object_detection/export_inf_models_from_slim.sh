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
pb_file_cnt=`ls -1 $ckt_path/$model_name/*/frozen_inference_graph.pb 2>/dev/null | wc -l`

inference_graph_name=$model_name$model_postfix
if [ $pb_file_cnt != 0 ]; then
  echo "frozen pb file found"
  mv $ckt_path/$model_name/*/frozen_inference_graph.pb  $frozen_inf_path/$inference_graph_name
else
  echo "no pb file found, return directly, no frozen pb generated!!!"
  return
fi

echo "all frozen fraph is stored at " $frozen_inf_path

echo switch back to original directory: $cur_path
cd $cur_path
