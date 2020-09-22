from collections import OrderedDict 
import json
import datetime
import argparse
'''
        self._cycles=["Start pod", 
            "Entering job preparation", 
            "Entering context manager injector", 
            "Start user script",
            "[Barrier]Start load CKPT",
            "End load CKPT",
            "Start 1st Iter/DataLoad",
            "Start run model on CPU",
            "Start exporting onnx",
            "In middle of building BW graph",
            "[Barrier]NCCL initialization",
            "NCCL INFO comm"]
'''
parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str)
args= parser.parse_args()

class EventSequence:
    def __init__(self):
        self._cycles=[
            "start pod", 
            "start job prepare", 
            "start cmi", 
            "start script",
            "start ckpt",
            "end ckpt",
            "start 1st ort step",
            "start exp onnx",
            #"in build bw",
            "NCCL init",
            "NCCL INFO comm"]

        self._event_id_name_map = OrderedDict()
        for i, s in enumerate(self._cycles):
            self._event_id_name_map[i] = s

    def map_event_string_to_id(self, event_string):
        return self._cycles.index(event_string)

    def map_event_id_to_string(self, event_id):
        return self._event_id_name_map[event_id]
    
    def sequence_length(self):
        return len(self._cycles)

event_seq = EventSequence()

class Event:
    def __init__(self):
        self.date_time = 0
        self.duration = -1

class Events:
    def __init__(self, event_id, is_global_event=False):
        self.is_global_event = is_global_event
        self._event_id = event_id
        self._events = OrderedDict()

    def __next__(self): # Python 2: def next(self)
        self.current += 1
        if self.current < self.high:
            return self.current
        raise StopIteration

    def update_date_time(self, date_time, rank=0):
        if rank not in self._events:
            #print("WARINING: Creating Event " + event_seq.map_event_id_to_string(self._event_id) + " for Rank " + str(rank) + ", date_time: " + str(date_time))
            self._events[rank] = Event()

        if self._events[rank].date_time != 0:
            pass
            #print("WARINING: Skipped to update date_time more than once for " + event_seq.map_event_id_to_string(self._event_id) + " on rank " + str(rank) + ", date_time: " + str(date_time))
        else:
            self._events[rank].date_time = date_time

    def update_duration(self, duration, rank=0):
        if rank not in self._events:
            #print("WARNING: Creating Event " + event_seq.map_event_id_to_string(self._event_id) + " for Rank " + str(rank))
            self._events[rank] = Event()

        if self._events[rank].duration != -1:
            pass
            #print("WARINING: Skipped to update duration more than once for " + event_seq.map_event_id_to_string(self._event_id) + " on rank " + str(rank))
        else:
            self._events[rank].duration = duration

    def get_event_by_rank(self, rank=0):
        if rank not in self._events:
            return None
        return self._events[rank]

    def __str__(self):
        s = "Events Logging: "
        for rank in self._events:
            s += " rank: " + str(rank) + " - date time: " + str(self._events[rank].date_time)
            s += " - duration: " + str(self._events[rank].duration)
        return s

class ResizeRecord:
    def __init__(self):
        self.world_size = -1
        self.phases = OrderedDict()
        for i in range(event_seq.sequence_length()):
            is_global_event = i < 3
            self.phases[i] = Events(i, is_global_event)

    def get_events(self, event_id):
        return self.phases[event_id]

f = open(args.log_dir, "r")
list_of_datetimes = []
values = []
lines = f.readlines()

resizes = []
index = 0
monitor_exit = True
for line in lines:
    elems = line.split(",")
    tag = elems[1].rstrip("\n")
    #print(elems)
    event_id = event_seq.map_event_string_to_id(tag)
    if event_id < 3:
        if monitor_exit:
            resizes.append(ResizeRecord())
            resize_record = resizes[index]
            #print("=============================Resize Event Seperator ===================================")
            monitor_exit = False
            index += 1

        happen_time = datetime.datetime.strptime(elems[0], "%Y-%m-%d %H:%M:%S")
        events_instance = resize_record.get_events(event_id)
        events_instance.update_date_time(happen_time, 0)

    else:
        monitor_exit = True
        rank = int(elems[2].rstrip("\n"))
        happen_time = datetime.datetime.strptime(elems[0], "%Y-%m-%d %H:%M:%S")
        events_instance = resize_record.get_events(event_id)
        events_instance.update_date_time(happen_time, rank)
        if tag == "NCCL INFO comm":
            resize_record.world_size = int(elems[3].rstrip("\n"))

for resize_record in resizes:
    for event_id in resize_record.phases:
        previous_event_id = event_id - 1
        if previous_event_id == -1:
            continue
        p_events = resize_record.get_events(previous_event_id)
        cur_events = resize_record.get_events(event_id)

        for rank in cur_events._events:
            event_instance = cur_events.get_event_by_rank(rank)
            p_rank_to_diff = rank if p_events.is_global_event is False else 0
            p_event_instance = p_events.get_event_by_rank(p_rank_to_diff)
            if p_event_instance is None:
                #print("Skip setting duration since previous date time did not exist. previous rank " 
                #      + str(p_rank_to_diff) + ", previous event: " + event_seq.map_event_id_to_string(previous_event_id)
                #      + ", current rank: " + str(rank))
                continue
            duration = (event_instance.date_time - p_event_instance.date_time).total_seconds()
            cur_events.update_duration(duration, rank)

f = open(args.log_dir + "_result.csv", "a")

for resize_record in resizes:
    title = ""
    i = 0
    for event_id in resize_record.phases:
        cur_events = resize_record.get_events(event_id)
        if i != 0:
            title += "duration,range,sample count,"
        title += "[" + event_seq.map_event_id_to_string(event_id) + "],"
        i += 1
    break
f.write(title + "\n")

for resize_record in resizes:
    s = ""
    i = 0
    for event_id in resize_record.phases:
        cur_events = resize_record.get_events(event_id)
        min_duration = 9999999
        max_duration = -9999999
        duration_sum = 0
        sample_count = 0
        for rank in cur_events._events:
            event_instance = cur_events.get_event_by_rank(rank)
            if event_instance.duration == -1:
                continue
            sample_count += 1
            min_duration = min(min_duration, event_instance.duration)
            max_duration = max(max_duration, event_instance.duration)
            duration_sum += event_instance.duration

        if i != 0:
            avg = str(duration_sum/sample_count) if sample_count > 0 else "N/A"
            s += avg + ","+ str(min_duration) + "-" + str(max_duration) + "," + str(sample_count) +","
        
        rank0_event = cur_events.get_event_by_rank(0)
        date_time = ""
        if rank0_event:
            date_time = str(rank0_event.date_time)
        s += date_time + ","
        i += 1

    f.write(s + "," + str(resize_record.world_size) + "\n")
f.close()
