import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from scipy.io import wavfile
import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import ddsp
from ddsp.training.encoders import Encoder
from ddsp.training.decoders import Decoder
from ddsp.training.models import Autoencoder
from ddsp.processors import Processor
import ddsp.training.nn as nn
from ddsp.core import exp_sigmoid


class ResidualLayer1D(tfkl.Layer):
    """A single layer for 1D ResNet, with a bottleneck (adapted from ddsp.training.nn)."""

    def __init__(self, ch, stride, shortcut, normalize, name=None):
        """Downsample waveform by stride, upsample channels by 2."""
        super().__init__(name=name)
        ch_out = 2 * ch
        self.shortcut = shortcut
        self.normalize = normalize

        # Layers.
        self.norm_input = self.normalize()
        if self.shortcut:
            self.conv_proj = tfkl.Conv1D(ch_out, 1, stride, padding="same", name="conv_proj")
        layers = [
            tfkl.Conv1D(ch, 1, 1, padding="same", name="conv1d"),
            self.build_norm_relu_conv(ch, 3, stride, name="norm_conv_relu_0"),
            self.build_norm_relu_conv(ch_out, 1, 1, name="norm_conv_relu_1"),
        ]
        self.bottleneck = tf.keras.Sequential(layers, name="bottleneck")

    def build_norm_relu_conv(self, ch, k, s, name="norm_relu_conv"):
        """Downsample waveform by stride."""
        layers = [
            self.normalize(),
            tfkl.Activation(tf.nn.relu),
            tfkl.Conv1D(ch, k, s, padding="same", name="conv1d"),
        ]
        return tf.keras.Sequential(layers, name=name)

    def call(self, x):
        r = x
        x = tf.nn.relu(self.norm_input(x))
        # The projection shortcut should come after the first norm and ReLU
        # since it performs a 1x1 convolution.
        r = self.conv_proj(x) if self.shortcut else r
        x = self.bottleneck(x)
        return x + r


def residual_stack1d(filters, block_sizes, strides, normalize, name="residual_stack"):
    """ResNet layers (adapted from ddsp.training.nn)."""
    layers = []
    for (ch, n_layers, stride) in zip(filters, block_sizes, strides):
        # Only the first block per residual_stack uses shortcut and strides.
        layers.append(ResidualLayer1D(ch, stride, True, normalize))
        # Add the additional (n_layers - 1) layers to the stack.
        for _ in range(1, n_layers):
            layers.append(ResidualLayer1D(ch, 1, False, normalize))
    layers.append(normalize())
    layers.append(tfkl.Activation(tf.nn.relu))
    return tf.keras.Sequential(layers, name=name)


def resnet1d(size="large", norm_type="layer", name="resnet"):
    """Residual network (adapted from ddsp.training.nn)."""
    size_dict = {
        "small": (32, [2, 3, 4]),
        "medium": (32, [3, 4, 6]),
        "large": (64, [3, 4, 6]),
    }
    ch, blocks = size_dict[size]

    if norm_type == "layer":
        normalize = tf.keras.layers.LayerNormalization
    elif norm_type == "group":
        normalize = lambda num_groups=32: tfa.layers.GroupNormalization(num_groups)
    else:
        normalize = tfa.layers.InstanceNormalization

    layers = [
        tfkl.Conv1D(64, 7, 2, padding="same", name="conv1d"),
        tfkl.MaxPool1D(pool_size=3, strides=2, padding="same"),
        residual_stack1d([ch, 2 * ch, 4 * ch], blocks, [1, 2, 2], normalize, name="residual_stack_0"),
        residual_stack1d([8 * ch, 8 * ch, 8 * ch], [3, 3, 3], [2, 2, 2], normalize, name="residual_stack_1"),
    ]
    return tf.keras.Sequential(layers, name=name)


class GrainEncoder(Encoder):
    """
    Residual temporal convolution variational grain encoder
    https://arxiv.org/abs/2008.01393
    """

    def __init__(self, grain_size=1024, overlap=0.75, latent_dim=96, resnet_size="large", name="grain_encoder"):
        super().__init__(name=name)

        self.grain_size = grain_size
        self.overlap = overlap
        self.latent_dim = latent_dim

        self.resnet = resnet1d(size=resnet_size)
        self.flat_size = self._flat_size()
        self.fc = nn.fc(self.flat_size)

        self.fc_mu = nn.fc(self.latent_dim)
        self.fc_logvar = nn.fc(self.latent_dim)

    def _flat_size(self):
        output = self.resnet(tf.ones((1, self.grain_size, 1)))
        return int(np.prod(output.shape[1:]))

    def compute_z(self, conditioning):
        """Takes in conditioning dictionary, returns a latent tensor z."""
        audio = conditioning["audio"]
        batch, n_samples, unit = audio.shape

        # extract overlapping grains
        stride = self.grain_size * (1 - self.overlap)
        grain_sequence = tf.image.extract_patches(
            audio[..., None],
            sizes=[1, self.grain_size, 1, 1],
            strides=[1, stride, 1, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )
        num_grains = grain_sequence.shape[1]
        grain_sequence = tf.reshape(grain_sequence, [batch * num_grains, self.grain_size, 1])

        # TODO make sure overlapping grain_envelopes sum to 1 at all times?
        grain_envelope = tf.concat(
            [
                tf.cast(tf.linspace(0, 1, int(self.grain_size / 4)), tf.float32),
                tf.ones((int(self.grain_size / 2)), dtype=tf.float32),
                tf.cast(tf.linspace(1, 0, int(self.grain_size / 4)), tf.float32),
            ],
            axis=0,
        )[None, :, None]
        grain_sequence = tf.multiply(grain_sequence, grain_envelope)

        h = self.resnet(grain_sequence)
        h2 = self.fc(tf.reshape(h, [batch * num_grains, self.flat_size]))
        mu = self.fc_mu(h2)
        logvar = self.fc_logvar(h2)

        sigma = tf.sqrt(tf.exp(logvar))
        eps = tfp.distributions.Normal(0, 1).sample(sample_shape=sigma.shape)
        z = mu + sigma * eps  # reparamaterization trick

        z_seq = tf.reshape(z, [batch, num_grains, self.latent_dim])
        return z_seq


class GrainDecoder(Decoder):
    """
    Spectral filtering decoder (adapted from ddsp.training.decoder)
    https://arxiv.org/abs/2008.01393
    """

    def __init__(
        self,
        ch=512,
        layers_per_stack=3,
        input_keys=("z",),
        output_splits=(("noise_magnitudes", 513),),  # grain_size / 2 + 1
        name="grain_decoder",
    ):
        super().__init__(output_splits=output_splits, name=name)
        stack = lambda: nn.fc_stack(ch, layers_per_stack)
        self.input_keys = input_keys

        self.input_stacks = [stack() for k in self.input_keys]
        self.out_stack = stack()
        self.dense_out = nn.dense(self.n_out)

    def decode(self, conditioning):
        """Takes in conditioning dictionary, returns dictionary of signals."""
        inputs = [conditioning[k] for k in self.input_keys]
        x = [stack(x) for stack, x in zip(self.input_stacks, inputs)]
        x = tf.concat(inputs + x, axis=-1)
        x = self.out_stack(x)
        return self.dense_out(x)


class NeuroGranular(Autoencoder):
    def __init__(self, sample_rate, example_secs, embedding_loss=False, name="granular"):
        encoder = GrainEncoder()
        decoder = GrainDecoder()

        noise = ddsp.synths.FilteredNoise(n_samples=example_secs * sample_rate, window_size=0, scale_fn=exp_sigmoid)
        processor_group = ddsp.processors.ProcessorGroup(dag=[(noise, ["noise_magnitudes"])])

        losses = [
            ddsp.losses.SpectralLoss(
                loss_type="L1", fft_sizes=(1024, 512, 256, 128, 64, 32), mag_weight=1.0, logmag_weight=1.0
            )
        ]
        if embedding_loss:
            losses += [ddsp.losses.PretrainedCREPEEmbeddingLoss(weight=0.1)]

        super(NeuroGranular, self).__init__(
            preprocessor=None,
            encoder=encoder,
            decoder=decoder,
            processor_group=processor_group,
            losses=losses,
            name=name,
        )

    def call(self, features, training=True):
        """Run the core of the network, get predictions and loss."""
        conditioning = self.encode(features, training=training)
        audio_grains = self.decode(conditioning, training=training)

        batch, num_grains, grain_size = audio_grains.shape

        # sum grains with overlap into full waveform
        stride = grain_size * (1 - self.overlap)
        signal = tf.zeros((batch, num_grains * stride))
        for i, grain in enumerate(grains):
            signal[i * stride : (i + 1) * stride] += grain  # TODO do grains need to be normalized to always sum to 1?

        # multi-channel temporal convolution that learns a parallel set of time-invariant FIR filters
        # and improves the audio quality of the assembled signal
        audio_gen = tfkl.Conv1D(64, 512, 1, padding="same", name="postprocess")(signal)

        if training:
            for loss_obj in self.loss_objs:
                loss = loss_obj(features["audio"], audio_gen)
                self._losses_dict[loss_obj.name] = loss
        return audio_gen


if __name__ == "__main__":
    sr, audio = wavfile.read("/home/hans/datasets/music-samples/test3/fedefadaeaSSURBEEALIKFFGARHONPRU.wav")
    audio = audio[None, : int(audio.shape[0] // 1024) * 1024, None].astype(np.float32)
    audio = audio / np.max([audio.max(), np.abs(audio.min())])
    audio = np.concatenate([audio] * 40, axis=0)
    print("audio shape: ", audio.shape)

    encoder = GrainEncoder()
    z = encoder.compute_z({"audio": audio})
    print("latent shape: ", z.shape)

    decoder = GrainDecoder()
    controls = decoder.decode({"z": z})
    print("controls shape: ", controls.shape)

    noise = ddsp.synths.FilteredNoise(n_samples=40960, window_size=0, scale_fn=exp_sigmoid)
    processor_group = ddsp.processors.ProcessorGroup(
        dag=[
            (noise, ["noise_magnitudes"]),
            (OverlapAdd(), ["filtered_noise/signal"]),
            (ddsp.effects.FIRFilter(window_size=512, trainable=True), ["overlap_add/signal"]),
        ]
    )
    output = processor_group(controls)
    print("output: ", output.shape)

