import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
import tensorflow_transform as tft

import ddsp


class Distortion(ddsp.processors.Processor):
    """Soft clipping distortion with Sigmoid function."""

    def __init__(self, scale_fn=ddsp.core.exp_sigmoid, trainable=False, name="distortion"):
        super().__init__(name=name, trainable=trainable)
        self.scale_fn = scale_fn

    def get_controls(self, audio, gain):
        """Convert network outputs into valid gain values.

        Args:
            audio: Dry audio. 2-D Tensor of shape [batch, n_samples].
            gain: Factor to multiply signal by (into sigmoid). Shape [batch_size, n_frames, 1].

        Returns:
            controls: Dictionary of tensors of synthesizer controls.
        """
        gain = ddsp.core.resample(gain[..., 0], audio.shape[-1])
        if self.scale_fn is not None:
            gain = self.scale_fn(gain)
        return {"audio": audio, "gain": gain}

    def get_signal(self, audio, gain):
        """Distort audio.

        Args:
            audio: Dry audio. 2-D Tensor of shape [batch, n_samples].
            gain: Factor to multiply signal by (into sigmoid). Shape [batch_size, 1].

        Returns:
            signal: Modulated audio of shape [batch, n_samples].
        """
        audio_out = tf.sigmoid((gain + 1e-1) * audio)
        audio_out = tf.subtract(audio_out, tf.reduce_min(audio_out))
        audio_out = tf.divide(audio_out, tf.reduce_max(audio_out))
        audio_out = tf.multiply(audio_out, 2)
        audio_out = tf.subtract(audio_out, 1)
        return audio_out


# ======== TESTS ========


def impulse_response_test():
    # show impulse response of distortion with different hardness
    x = np.linspace(-1.5, 1.5, 1000, dtype=np.float32)
    for gain in range(0, 21, 2):
        audio_gen = Distortion(scale_fn=None)(x, gain * np.ones((1, 1), dtype=np.float32)).numpy().squeeze()
        plt.plot(x, audio_gen, label=str(gain))
    plt.legend()
    plt.show()
    plt.close()


def audio_test(audio_file, sample_rate):
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
        wavfile.write("output/" + output_name, sample_rate, array_of_ints)


if __name__ == "__main__":
    import argparse
    import numpy as np
    import librosa.display
    from scipy.io import wavfile
    import matplotlib.pyplot as plt
    from ddsp.colab.colab_utils import audio_bytes_to_np

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    impulse_response_test()
    audio_test(args.audio, args.sr)
