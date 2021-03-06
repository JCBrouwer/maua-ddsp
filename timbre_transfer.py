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

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
import copy
import time
import glob
import argparse

import ddsp
import ddsp.training
from ddsp.colab.colab_utils import (
    auto_tune,
    detect_notes,
    fit_quantile_transform,
    get_tuning_factor,
    audio_bytes_to_np,
    DEFAULT_SAMPLE_RATE,
)
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import pickle

import models

parser = argparse.ArgumentParser()
parser.add_argument("--audio", type=str, required=True)
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--model", type=str, required=True, choices=["gin", "Default", "SimpleDistortion", "NeuroBass"])
parser.add_argument("--data_stats", type=str)
parser.add_argument("--sr", type=int, default=DEFAULT_SAMPLE_RATE)
parser.add_argument("--example_secs", type=int, default=4)
parser.add_argument("--no_adjust", action="store_true")
parser.add_argument("--threshold", type=float, default=1)  # 0 - 2
parser.add_argument("--quiet", type=float, default=30)  # 0 - 60
parser.add_argument("--autotune", type=float, default=0)  # 0 - 1
parser.add_argument("--pitch_shift", type=int, default=0)  # -2 - 2
parser.add_argument("--loudness_shift", type=float, default=0)  # -20 - 20
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()

with open(args.audio, "rb") as f:
    audio = audio_bytes_to_np(f.read(), sample_rate=args.sr)
audio = audio[np.newaxis, :]
print("\nExtracting audio features...")

# Setup the session.
ddsp.spectral_ops.reset_crepe()

# Compute features.
start_time = time.time()
audio_features = ddsp.training.metrics.compute_audio_features(audio)
audio_features["loudness_db"] = audio_features["loudness_db"].astype(np.float32)
audio_features_mod = None
print("Audio features took %.1f seconds" % (time.time() - start_time))

trim = -15

# Iterate through directories until model directory is found
for root, dirs, filenames in os.walk("/".join(args.ckpt.split("/")[:-1])):
    for filename in filenames:
        if filename.endswith(".gin") and not filename.startswith("."):
            model_dir = root
            gin_file = os.path.join(model_dir, filename)
            break

# Load the dataset statistics.
dataset_stats = None
if args.data_stats is not None:
    print(f"Loading dataset statistics from {args.data_stats}")
    with open(args.data_stats, "rb") as f:
        dataset_stats = pickle.load(f)

if args.model == "gin":
    import gin

    # Parse gin config,
    with gin.unlock_config():
        gin.parse_config_file(gin_file, skip_unknown=True)

    time_steps_train = gin.query_parameter("DefaultPreprocessor.time_steps")
    n_samples_train = gin.query_parameter("Additive.n_samples")
    hop_size = int(n_samples_train / time_steps_train)
    time_steps = int(audio.shape[1] / hop_size)
    n_samples = time_steps * hop_size

    print()
    print("===Trained model===")
    print("Time Steps", time_steps_train)
    print("Samples", n_samples_train)
    print("Hop Size", hop_size)
    print("\n===Resynthesis===")
    print("Time Steps", time_steps)
    print("Samples", n_samples)
    print()

    gin_params = [
        "Additive.n_samples = {}".format(n_samples),
        "FilteredNoise.n_samples = {}".format(n_samples),
        "DefaultPreprocessor.time_steps = {}".format(time_steps),
        "oscillator_bank.use_angular_cumsum = True",  # Avoids cumsum accumulation errors.
    ]

    with gin.unlock_config():
        gin.parse_config(gin_params)

    # Trim all input vectors to correct lengths
    for key in ["f0_hz", "f0_confidence", "loudness_db"]:
        audio_features[key] = audio_features[key][:time_steps]
    audio_features["audio"] = audio_features["audio"][:, :n_samples]

    # Set up the model just to predict audio given new conditioning
    print("Loading: ", args.ckpt)
    model = ddsp.training.models.Autoencoder()
    model.restore(args.ckpt)
else:
    print("Loading: ", args.ckpt)
    model = models.__dict__[args.model](sample_rate=args.sr, example_secs=args.example_secs)
    model.restore(args.ckpt)

    # Ensure dimensions and sampling rates are equal
    time_steps_train = model.preprocessor.time_steps
    n_samples_train = args.example_secs * args.sr
    hop_size = int(n_samples_train / time_steps_train)
    time_steps = int(audio.shape[1] / hop_size)
    n_samples = time_steps * hop_size

    print()
    print("===Trained model===")
    print("Time Steps", time_steps_train)
    print("Samples", n_samples_train)
    print("Hop Size", hop_size)
    print("\n===Resynthesis===")
    print("Time Steps", time_steps)
    print("Samples", n_samples)
    print()

    # Trim all input vectors to correct lengths
    for key in ["f0_hz", "f0_confidence", "loudness_db"]:
        audio_features[key] = audio_features[key][:time_steps]
    audio_features["audio"] = audio_features["audio"][:, :n_samples]

# Build model by running a batch through it.
start_time = time.time()
_ = model(audio_features, training=False)
print("Restoring model took %.1f seconds" % (time.time() - start_time))

audio_features_mod = {k: v.copy() for k, v in audio_features.items()}

## Helper functions.
def shift_ld(audio_features, ld_shift=0.0):
    """Shift loudness by a number of ocatves."""
    audio_features["loudness_db"] += ld_shift
    return audio_features


def shift_f0(audio_features, pitch_shift=0.0):
    """Shift f0 by a number of ocatves."""
    audio_features["f0_hz"] *= 2.0 ** (pitch_shift)
    audio_features["f0_hz"] = np.clip(audio_features["f0_hz"], 0.0, librosa.midi_to_hz(110.0))
    return audio_features


mask_on = None

if not args.no_adjust and dataset_stats is not None:
    # Detect sections that are "on".
    mask_on, note_on_value = detect_notes(
        audio_features["loudness_db"], audio_features["f0_confidence"], args.threshold
    )

    if np.any(mask_on):
        # Shift the pitch register.
        target_mean_pitch = dataset_stats["mean_pitch"]
        pitch = ddsp.core.hz_to_midi(audio_features["f0_hz"])
        mean_pitch = np.mean(pitch[mask_on])
        p_diff = target_mean_pitch - mean_pitch
        p_diff_octave = p_diff / 12.0
        round_fn = np.floor if p_diff_octave > 1.5 else np.ceil
        p_diff_octave = round_fn(p_diff_octave)
        audio_features_mod = shift_f0(audio_features_mod, p_diff_octave)

        # Quantile shift the note_on parts.
        _, loudness_norm = fit_quantile_transform(
            audio_features["loudness_db"], mask_on, inv_quantile=dataset_stats["quantile_transform"]
        )

        # Turn down the note_off parts.
        mask_off = np.logical_not(mask_on)
        loudness_norm[mask_off] -= args.quiet * (1.0 - note_on_value[mask_off][:, np.newaxis])
        loudness_norm = np.reshape(loudness_norm, audio_features["loudness_db"].shape)

        audio_features_mod["loudness_db"] = loudness_norm

        # Auto-tune.
        if args.autotune:
            f0_midi = np.array(ddsp.core.hz_to_midi(audio_features_mod["f0_hz"]))
            tuning_factor = get_tuning_factor(f0_midi, audio_features_mod["f0_confidence"], mask_on)
            f0_midi_at = auto_tune(f0_midi, tuning_factor, mask_on, amount=args.autotune)
            audio_features_mod["f0_hz"] = ddsp.core.midi_to_hz(f0_midi_at)

    else:
        print("\nSkipping auto-adjust (no notes detected or --adjust not specified).")

else:
    print("\nSkipping auto-adjust (--adjust not specified or no dataset statistics found).")

# Manual Shifts.
audio_features_mod = shift_ld(audio_features_mod, args.loudness_shift)
audio_features_mod = shift_f0(audio_features_mod, args.pitch_shift)

af = audio_features if audio_features_mod is None else audio_features_mod

# Run a batch of predictions.
start_time = time.time()
audio_gen = model(af, training=False)
print("Prediction took %.1f seconds" % (time.time() - start_time))

print("Resynthesis")
# If batched, take first element.
if len(audio_gen.shape) == 2:
    audio_gen = audio_gen[0]
audio_gen = np.asarray(audio_gen)

# Plot Features.
if args.plot:
    fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True, figsize=(18, 8))
    ax[0, 0].plot(audio_features["loudness_db"][:trim])
    ax[0, 0].set_ylabel("loudness_db")

    ax[1, 0].plot(librosa.hz_to_midi(audio_features["f0_hz"][:trim]))
    ax[1, 0].set_ylabel("f0 [midi]")

    ax[2, 0].plot(audio_features["f0_confidence"][:trim])
    ax[2, 0].set_ylabel("f0 confidence")
    _ = ax[2, 0].set_xlabel("Time step [frame]")

    has_mask = int(mask_on is not None)

    ax[0, 1].plot(audio_features["loudness_db"][:trim])
    ax[0, 1].plot(audio_features_mod["loudness_db"][:trim])
    ax[0, 1].set_ylabel("loudness_db")
    ax[0, 1].legend(["Original", "Adjusted"])

    if has_mask:
        ax[1, 1].plot(np.ones_like(mask_on[:trim]) * args.threshold, "k:")
        ax[1, 1].plot(note_on_value[:trim])
        ax[1, 1].plot(mask_on[:trim])
        ax[1, 1].set_ylabel("Note-on Mask")
        ax[1, 1].set_xlabel("Time step [frame]")
        ax[1, 1].legend(["Threshold", "Likelihood", "Mask"])

    ax[1 + has_mask, 1].plot(librosa.hz_to_midi(audio_features["f0_hz"][:trim]))
    ax[1 + has_mask, 1].plot(librosa.hz_to_midi(audio_features_mod["f0_hz"][:trim]))
    ax[1 + has_mask, 1].set_ylabel("f0 [midi]")
    _ = ax[1 + has_mask, 1].legend(["Original", "Adjusted"])

    librosa.display.specshow(
        librosa.power_to_db(librosa.feature.melspectrogram(y=audio.squeeze(), sr=args.sr), ref=np.max), ax=ax[0, 2]
    )
    librosa.display.specshow(
        librosa.power_to_db(librosa.feature.melspectrogram(y=audio_gen, sr=args.sr), ref=np.max), ax=ax[1, 2]
    )
    plt.show()
    plt.close()

normalizer = float(np.iinfo(np.int16).max)
array_of_ints = np.array(audio_gen * normalizer, dtype=np.int16)

output_name = "_".join(
    [
        args.audio.split("/")[-1].split(".")[0],
        args.ckpt.split("/")[-2],
        args.ckpt.split("/")[-1].replace("ckpt-", ""),
        f"thresh{args.threshold:.2f}",
        f"quiet{int(args.quiet)}",
        f"tune{args.autotune:.2f}",
        f"pitch{int(args.pitch_shift)}",
        f"loud{int(args.loudness_shift)}",
        ".wav",
    ]
)
wavfile.write("output/" + output_name, args.sr, array_of_ints)

