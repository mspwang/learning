## Run on Single Node


	python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=$MASTER_IP --master_port=29500 benchmark_models.py -n 50 -b 4


## Run on Multi Nodes


On each node, manually run:
	python -m torch.distributed.launch --nproc_per_node=8 --nnodes=<node_count> --node_rank=<node_rank> --master_addr=$MASTER_IP --master_port=29500 benchmark_models.py -n 50 -b 4


Or use mpirun:

	mpirun --tag-output -hostfile /job/${DLTS_JOB_ID}/hostfile --npernode 1 -bind-to none -x MASTER_IP -x NCCL_DEBUG=INFO -x NCCL_TREE_THRESHOLD=0 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0 bash dist_cmd.sh


