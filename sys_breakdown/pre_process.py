import os
import re
import datetime
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str)
args= parser.parse_args()


f = open(args.log_dir, "r")
o = open(args.log_dir + "_processed", "w")
lines = f.readlines()
previous_start_monitor_time = None
t=None
s=None
for line in lines:
    regexp="[\s\S]+(2020-[0-9]+-[0-9]+ [0-9-:]+) (start monitoring log from pod)"
    match = re.match(regexp, line)
    if match:
        date_time=str(match.group(1))
        tag=str(match.group(2))
        tag="start pod"
        date_time = datetime.datetime.strptime(date_time,"%Y-%m-%d %H:%M:%S")
        if previous_start_monitor_time is None:
            previous_start_monitor_time = date_time
        #o.writelines([str((date_time - previous_start_monitor_time).seconds) + "," + tag + "\n"])
        o.writelines([str(date_time) + "," + tag + "\n"])
        continue
    regexp="([0-9-:]+ [0-9-:]+) [\s\S]+ (Entering job preparation)."
    match = re.match(regexp, line)
    if match:
        date_time=str(match.group(1))
        tag=str(match.group(2))
        tag="start job prepare"
        date_time = datetime.datetime.strptime(date_time,"%Y-%m-%d %H:%M:%S")
        #o.writelines([str((date_time - previous_start_monitor_time).seconds) + "," + tag + "\n"])
        o.writelines([str(date_time) + "," + tag + "\n"])
        continue

    regexp="([0-9-:]+ [0-9-:]+) [\s\S]+ \[1,([0-9]+)\][\s\S]+ (Entering context manager injector)."
    match = re.match(regexp, line)
    if match:
        date_time=str(match.group(1))
        rank=str(match.group(2))
        tag=str(match.group(3))
        # Ignore this pass, because the name is confusion with below pattern
        # date_time = datetime.datetime.strptime(date_time,"%Y-%m-%d %H:%M:%S")
        # #o.writelines([str((date_time - previous_start_monitor_time).seconds) + "," + tag + "," + rank + "\n"])
        # o.writelines([str(date_time) + "," + tag + "," + str(rank) +"\n"])
        continue

    regexp="([0-9-:]+ [0-9-:]+) [\s\S]+[1,][\s\S]+ (Entering context manager injector)."
    match = re.match(regexp, line)
    if match:
        date_time=str(match.group(1))
        tag=str(match.group(2))
        tag="start cmi"
        date_time = datetime.datetime.strptime(date_time,"%Y-%m-%d %H:%M:%S")
        #o.writelines([str((date_time - previous_start_monitor_time).seconds) + "," + tag + "\n"])
        o.writelines([str(date_time) + "," + tag + "\n"])
        continue

    regexp="([0-9-:]+ [0-9-:]+) [\s\S]+ \[1,([0-9]+)\][\s\S]+======================(in run_pretraining_ort.py)=================="
    match = re.match(regexp, line)
    if match:
        date_time=str(match.group(1))
        rank=str(match.group(2))
        #tag=str(match.group(3))
        tag="start script"
        date_time = datetime.datetime.strptime(date_time,"%Y-%m-%d %H:%M:%S")
        #o.writelines([str((date_time - previous_start_monitor_time).seconds) + "," + tag + "," + rank + "\n"])
        o.writelines([str(date_time) + "," + tag + "," + str(rank) +"\n"])
        continue

    regexp="([0-9-:]+ [0-9-:]+) [\s\S]+ \[1,([0-9]+)\][\s\S]+(resume global_step)[\s\S]+"
    match = re.match(regexp, line)
    if match:
        date_time=str(match.group(1))
        rank=str(match.group(2))
        tag=str(match.group(3))
        tag="start ckpt"
        date_time = datetime.datetime.strptime(date_time,"%Y-%m-%d %H:%M:%S")
        #o.writelines([str((date_time - previous_start_monitor_time).seconds) + "," + tag + "," + rank + "\n"])
        o.writelines([str(date_time) + "," + tag + "," + str(rank) +"\n"])
        continue

    regexp="([0-9-:]+ [0-9-:]+) [\s\S]+ \[1,([0-9]+)\][\s\S]+(after load checkpoint 2)[\s\S]+"
    match = re.match(regexp, line)
    if match:
        date_time=str(match.group(1))
        rank=str(match.group(2))
        tag=str(match.group(3))
        tag="end ckpt"
        date_time = datetime.datetime.strptime(date_time,"%Y-%m-%d %H:%M:%S")
        #o.writelines([str((date_time - previous_start_monitor_time).seconds) + "," + tag + "," + rank + "\n"])
        o.writelines([str(date_time) + "," + tag + "," + str(rank) +"\n"])
        continue

    regexp="([0-9-:]+ [0-9-:]+) [\s\S]+ \[1,([0-9]+)\][\s\S]+(current worker use file id)[\s\S]+"
    match = re.match(regexp, line)
    if match:
        date_time=str(match.group(1))
        rank=str(match.group(2))
        tag=str(match.group(3))
        tag="start 1st ort step"
        date_time = datetime.datetime.strptime(date_time,"%Y-%m-%d %H:%M:%S")
        #o.writelines([str((date_time - previous_start_monitor_time).seconds) + "," + tag + "," + rank + "\n"])
        o.writelines([str(date_time) + "," + tag + "," + str(rank) + "\n"])
        continue

    # regexp="([0-9-:]+ [0-9-:]+) [\s\S]+ \[1,([0-9]+)\][\s\S]+(BertPretrainingCriterion: batch_size:  [0-9]+) , self.seq_length: [0-9]+"
    # match = re.match(regexp, line)
    # if match:
    #     date_time=str(match.group(1))
    #     rank=str(match.group(2))
    #     tag=str(match.group(3))
    #     tag="start cpu run"
    #     date_time = datetime.datetime.strptime(date_time,"%Y-%m-%d %H:%M:%S")
    #     #o.writelines([str((date_time - previous_start_monitor_time).seconds) + "," + tag + "," + rank + "\n"])
    #     o.writelines([str(date_time) + "," + tag + "," + str(rank) + "\n"])
    #     continue

    regexp="([0-9-:]+ [0-9-:]+) [\s\S]+ \[1,([0-9]+)\][\s\S]+(Dropout is a training op and should not be exported in inference mode). Make sure to call eval\(\) on the model, and to export it with param training=False."
    match = re.match(regexp, line)
    if match:
        date_time=str(match.group(1))
        rank=str(match.group(2))
        tag=str(match.group(3))
        tag="start exp onnx"
        date_time = datetime.datetime.strptime(date_time,"%Y-%m-%d %H:%M:%S")
        #o.writelines([str((date_time - previous_start_monitor_time).seconds) + "," + tag + "," + rank + "\n"])
        o.writelines([str(date_time) + "," + tag + "," + str(rank) +"\n"])
        continue
    
    # regexp="([0-9-:]+ [0-9-:]+) [\s\S]+ \[1,([0-9]+)\][\s\S]+(Skip building gradient for node: Expand_)[\s\S]+"
    # match = re.match(regexp, line)
    # if match:
    #     date_time=str(match.group(1))
    #     rank=str(match.group(2))
    #     tag=str(match.group(3))
    #     tag="in build bw"
    #     date_time = datetime.datetime.strptime(date_time,"%Y-%m-%d %H:%M:%S")
    #     #o.writelines([str((date_time - previous_start_monitor_time).seconds) + "," + tag + "," + rank + "\n"])
    #     o.writelines([str(date_time) + "," + tag + "," + str(rank) +"\n"])
    #     continue
    
    regexp="([0-9-:]+ [0-9-:]+) [\s\S]+ \[1,([0-9]+)\][\s\S]+(NCCL INFO Bootstrap) : [\s\S]+"
    match = re.match(regexp, line)
    if match:
        date_time=str(match.group(1))
        rank=str(match.group(2))
        tag=str(match.group(3))
        tag="NCCL init"
        date_time = datetime.datetime.strptime(date_time,"%Y-%m-%d %H:%M:%S")
        #o.writelines([str((date_time - previous_start_monitor_time).seconds) + "," + tag + "," + rank + "\n"])
        o.writelines([str(date_time) + "," + tag + "," + str(rank) + "\n"])
        continue
    
    regexp="([0-9-:]+ [0-9-:]+) [\s\S]+ \[1,([0-9]+)\][\s\S]+(NCCL INFO comm)[\s\S]+ rank [0-9]+ nranks ([0-9]+) cudaDev [0-9]+ nvmlDev [0-9]+ - Init COMPLETE"
    match = re.match(regexp, line)
    if match:
        date_time=str(match.group(1))
        rank=str(match.group(2))
        tag=str(match.group(3))
        world=str(match.group(4))
        date_time = datetime.datetime.strptime(date_time,"%Y-%m-%d %H:%M:%S")
        #o.writelines([str((date_time - previous_start_monitor_time).seconds) + "," + tag + "," + rank + "\n"])
        o.writelines([str(date_time) + "," + tag + "," + str(rank) + "," + str(world) + "\n"])
        continue

o.close()
f.close()