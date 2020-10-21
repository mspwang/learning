
echo $MASTER_IP" is master ip"
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${OMPI_COMM_WORLD_SIZE} --node_rank=${OMPI_COMM_WORLD_RANK} --master_addr=$MASTER_IP --master_port=29500 benchmark_models.py -n 50 -b 4
