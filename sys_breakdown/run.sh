
filename=$1

#grep -e "Entering job preparation" -e "Entering context manager injector" -e "start monitoring log from pod" -e "in run_pretraining_ort.py" -e "resume global_step" -e "after load checkpoint 2" -e "current worker use file id " -e "UserWarning: Dropout" -e "Skip building gradient for node" -e "NCCL INFO " -e "Iteration: " speech_cluster_bert.txt
#grep --text -E "Entering job preparation|Entering context manager injector|start monitoring log from pod|in run_pretraining_ort.py|resume global_step|after load checkpoint 2|current worker use file id |UserWarning: Dropout|Skip building gradient for node|NCCL INFO |Iteration: |:BertPretrainingCriterion: batch_size" bench.txt

echo "filtering out useful logs into "$filename"_filter"
grep --text -E "Entering job preparation|Entering context manager injector|start monitoring log from pod|in run_pretraining_ort.py|resume global_step|after load checkpoint 2|current worker use file id |UserWarning: Dropout|Skip building gradient for node|:BertPretrainingCriterion: batch_size|NCCL INFO |Iteration: " $filename > $filename"_filter"

echo "pre-process log file into "$filename"_filter_processed"
python pre_process.py --log_dir=$filename"_filter"

echo "analyzing log file into "$filename"_filter_processed_result.csv"
python analyze.py --log_dir=$filename"_filter_processed"