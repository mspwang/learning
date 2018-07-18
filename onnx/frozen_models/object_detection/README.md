Those scripts are targeting exporting inference models from sources reported by https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md.

frozen inference models are already in the zip file provided in the repo, so we don't need do export effort unless the tf1.5 based frozen model did not meet our requirements.

download.sh is the trigger
