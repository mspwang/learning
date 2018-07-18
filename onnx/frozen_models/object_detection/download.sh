#/bin/bash

source shared_env

IFS="="
while read -r name host filename
do
  frozen_graph_path=$frozen_inf_path/$name$model_postfix 
  if [ ! -f $frozen_graph_path ]; then
    echo "$frozen_graph_path not found, so start generating..."
    bash export_inf_models_from_slim.sh "$name" "$host" "$filename" 
  else
    echo "Frozen model $frozen_graph_path already exists, quits"
  fi
done < inference_models.conf
