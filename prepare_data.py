# Copyright 2020 Google LLC. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import glob
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf

from ddsp.colab import colab_utils
import ddsp.training
from ddsp.training.data_preparation import prepare_tfrecord_lib

parser = argparse.ArgumentParser()
parser.add_argument("--audio_dir", type=str, required=True)
parser.add_argument("--save_dir", type=str, required=True)
parser.add_argument("--sr", type=int, default=16000)
parser.add_argument("--frame_rate", type=float, default=250)
parser.add_argument("--example_secs", type=float, default=4)
args = parser.parse_args()

audio_pattern = args.audio_dir + "/*"
os.makedirs(args.save_dir, exist_ok=True)
tfrecord = os.path.join(args.audio_dir, "train.tfrecord")
tfrecord_pattern = tfrecord + "*"

# Make a new dataset.
if not glob.glob(audio_pattern):
    raise ValueError("No audio files found. Please use the previous cell to upload.")

if not glob.glob(tfrecord_pattern):
    prepare_tfrecord_lib.prepare_tfrecord(
        glob.glob(audio_pattern),
        tfrecord,
        num_shards=16,
        sample_rate=args.sr,
        frame_rate=args.frame_rate,
        window_secs=args.example_secs,
        hop_secs=1,
        pipeline_options="",
    )

if not glob.glob(os.path.join(args.audio_dir, "dataset_statistics.pkl")):
    data_provider = ddsp.training.data.TFRecordProvider(
        tfrecord_pattern, example_secs=args.example_secs, frame_rate=args.frame_rate, sample_rate=args.sr,
    )
    dataset = data_provider.get_dataset(shuffle=False)
    colab_utils.save_dataset_statistics(data_provider, os.path.join(args.audio_dir, "dataset_statistics.pkl"))


data_provider = ddsp.training.data.TFRecordProvider(
    tfrecord_pattern, example_secs=args.example_secs, frame_rate=args.frame_rate, sample_rate=args.sr,
)
dataset = data_provider.get_dataset(shuffle=False)

try:
    ex = next(iter(dataset))
except StopIteration:
    raise ValueError(
        "TFRecord contains no examples. Please try re-running the pipeline with " "different audio file(s)."
    )
