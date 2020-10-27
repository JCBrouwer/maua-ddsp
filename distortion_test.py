import argparse

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf

import ddsp
from ddsp.colab.colab_utils import audio_bytes_to_np

import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from scipy.io import wavfile

from distortion import Distortion


def distortion_impulse_response():
    # show impulse response of distortion with different hardness
    x = np.linspace(-1.5, 1.5, 1000, dtype=np.float32)
    for gain in range(0, 21, 2):
        audio_gen = Distortion(scale_fn=None)(x, gain * np.ones((1, 1), dtype=np.float32)).numpy().squeeze()
        plt.plot(x, audio_gen, label=str(gain))
    plt.legend()
    plt.show()
    plt.close()


def distortion_test(audio_file, sample_rate):
    with open(audio_file, "rb") as f:
        audio = audio_bytes_to_np(f.read(), sample_rate=sample_rate)
    audio = audio[np.newaxis, :].astype(np.float32)

    for gain in range(0, 31, 5):
        # apply effect to supplied audio
        audio_gen = Distortion(scale_fn=None)(audio, gain * np.ones((1, 1), dtype=np.float32)).numpy().squeeze()

        # plot spectrograms before and after
        fig, ax = plt.subplots(2, 1)
        librosa.display.specshow(
            librosa.power_to_db(librosa.feature.melspectrogram(y=audio.squeeze(), sr=sample_rate), ref=np.max), ax=ax[0]
        )
        librosa.display.specshow(
            librosa.power_to_db(librosa.feature.melspectrogram(y=audio_gen, sr=sample_rate), ref=np.max), ax=ax[1]
        )
        plt.show()
        plt.close()

        # write processed wav
        normalizer = float(np.iinfo(np.int16).max)
        array_of_ints = np.array(audio_gen * normalizer, dtype=np.int16)
        output_name = audio_file.split("/")[-1].split(".")[0] + "_gain_" + str(gain) + ".wav"
        wavfile.write(output_name, sample_rate, array_of_ints)


if __name__ == "__main__":
    distortion_impulse_response()

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    distortion_test(args.audio, args.sr)
