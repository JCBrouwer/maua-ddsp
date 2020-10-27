import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_graphics.math.math_helpers import factorial
import ddsp


# def binomial_coefficient(n, k):
#     return factorial(n) / (factorial(k) * factorial(n - k))


class Distortion(ddsp.processors.Processor):
    """Soft clipping distortion with Sigmoid function."""

    # hardness: Controls smoothing of distrotion. 0 is pure digitial hard clipping, higher values give
    #     progressively steeper-sloped soft clipping. Beware: 6 and higher causes overflow errors.

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
        # audio = tf.keras.backend.clip(audio, -gain, gain)
        # audio = tf.divide(audio, gain)
        # audio = tf.add(audio, 1)
        # audio = tf.divide(audio, 2)

        # octaves = tf.range(0, self.hardness + 1, dtype=tf.float32)[:, None, None]
        # audio_out = tf.pow(-audio, octaves)
        # audio_out = tf.multiply(audio_out, binomial_coefficient(self.hardness + octaves, octaves))
        # audio_out = tf.multiply(audio_out, binomial_coefficient(2 * self.hardness + 1, self.hardness - octaves))
        # audio_out = tf.reduce_sum(audio_out, axis=0)
        # audio_out = tf.multiply(audio_out, tf.pow(audio, self.hardness + 1))

        audio_out = tf.sigmoid((gain + 1e-1) * audio)

        audio_out = tf.subtract(audio_out, tf.reduce_min(audio_out))
        audio_out = tf.divide(audio_out, tf.reduce_max(audio_out))
        audio_out = tf.multiply(audio_out, 2)
        audio_out = tf.subtract(audio_out, 1)

        return audio_out
