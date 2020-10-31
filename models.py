import ddsp
import ddsp.training
from ddsp.training.models import Autoencoder

from distortion import Distortion
from granular import NeuroGranular


class Default(Autoencoder):
    def __init__(
        self,
        sample_rate,
        example_secs,
        embedding_loss=False,
        preprocessor=None,
        encoder=None,
        decoder=None,
        processor_group=None,
        name="default",
    ):
        if preprocessor is None:
            preprocessor = ddsp.training.preprocessing.DefaultPreprocessor(time_steps=1000)

        if encoder is None:
            encoder = ddsp.training.encoders.MfccTimeDistributedRnnEncoder(
                rnn_channels=512, rnn_type="gru", z_dims=16, z_time_steps=125
            )

        if decoder is None:
            decoder = ddsp.training.decoders.RnnFcDecoder(
                rnn_channels=512,
                rnn_type="gru",
                ch=512,
                layers_per_stack=3,
                input_keys=("ld_scaled", "f0_scaled", "z"),
                output_splits=(("amps", 1), ("harmonic_distribution", 100), ("noise_magnitudes", 65)),
            )

        losses = [ddsp.losses.SpectralLoss(loss_type="L1", mag_weight=1.0, logmag_weight=0.1)]
        if embedding_loss:
            losses += [ddsp.losses.PretrainedCREPEEmbeddingLoss(weight=0.1)]

        if processor_group is None:
            additive = ddsp.synths.Additive(
                name="additive",
                n_samples=example_secs * sample_rate,
                sample_rate=sample_rate,
                normalize_below_nyquist=True,
                scale_fn=ddsp.core.exp_sigmoid,
            )
            noise = ddsp.synths.FilteredNoise(
                name="filtered_noise",
                n_samples=example_secs * sample_rate,
                window_size=0,
                scale_fn=ddsp.core.exp_sigmoid,
            )
            add = ddsp.processors.Add(name="add")
            processor_group = ddsp.processors.ProcessorGroup(
                dag=[
                    (additive, ["amps", "harmonic_distribution", "f0_hz"]),
                    (noise, ["noise_magnitudes"]),
                    (add, ["filtered_noise/signal", "additive/signal"]),
                ],
                name="processor_group",
            )

        super(Default, self).__init__(
            preprocessor=preprocessor,
            encoder=encoder,
            decoder=decoder,
            processor_group=processor_group,
            losses=losses,
            name=name,
        )


class SimpleDistortion(Default):
    def __init__(self, sample_rate, example_secs, embedding_loss=False):
        decoder = ddsp.training.decoders.RnnFcDecoder(
            rnn_channels=512,
            rnn_type="gru",
            ch=512,
            layers_per_stack=3,
            input_keys=("ld_scaled", "f0_scaled", "z"),
            output_splits=(
                ("amps", 1),
                ("harmonic_distribution", 100),
                ("noise_magnitudes", 65),
                ("distortion_threshold", 1),
            ),
        )

        processor_group = ddsp.processors.ProcessorGroup(
            dag=[
                (
                    ddsp.synths.Additive(n_samples=example_secs * sample_rate, sample_rate=sample_rate),
                    ["amps", "harmonic_distribution", "f0_hz"],
                ),
                (ddsp.synths.FilteredNoise(n_samples=example_secs * sample_rate, name="noise"), ["noise_magnitudes"]),
                (ddsp.processors.Add(name="sum"), ["noise/signal", "additive/signal"]),
                (Distortion(trainable=True, name="distortion"), ["sum/signal", "distortion_threshold"]),
            ],
            name="processor_group",
        )

        super(SimpleDistortion, self).__init__(
            sample_rate=sample_rate,
            example_secs=example_secs,
            embedding_loss=embedding_loss,
            decoder=decoder,
            processor_group=processor_group,
            name="distortion",
        )


class NeuroBass(Default):
    def __init__(self, sample_rate, example_secs, embedding_loss=False):
        decoder = ddsp.training.decoders.RnnFcDecoder(
            rnn_channels=512,
            rnn_type="gru",
            ch=512,
            layers_per_stack=3,
            input_keys=("ld_scaled", "f0_scaled", "z"),
            output_splits=(
                ("amps", 1),
                ("harmonic_distribution", 100),
                ("noise_magnitudes1", 65),
                ("noise_magnitudes2", 65),
                ("filter_magnitudes1", 65),
                ("filter_magnitudes2", 65),
                ("distortion_threshold1", 1),
                ("distortion_threshold2", 1),
            ),
        )

        processor_group = ddsp.processors.ProcessorGroup(
            dag=[
                (
                    ddsp.synths.Additive(n_samples=example_secs * sample_rate, sample_rate=sample_rate),
                    ["amps", "harmonic_distribution", "f0_hz"],
                ),
                (ddsp.synths.FilteredNoise(n_samples=example_secs * sample_rate, name="noise1"), ["noise_magnitudes1"]),
                (ddsp.processors.Add(name="sum1"), ["noise1/signal", "additive/signal"]),
                (ddsp.effects.FIRFilter(name="filter1"), ["sum1/signal", "filter_magnitudes1"]),
                (Distortion(trainable=True, name="distortion1"), ["filter1/signal", "distortion_threshold1"]),
                (ddsp.synths.FilteredNoise(n_samples=example_secs * sample_rate, name="noise2"), ["noise_magnitudes2"]),
                (ddsp.processors.Add(name="sum2"), ["noise2/signal", "distortion1/signal"]),
                (ddsp.effects.FIRFilter(name="filter2"), ["sum2/signal", "filter_magnitudes2"]),
                (Distortion(trainable=True, name="distortion2"), ["filter2/signal", "distortion_threshold2"]),
                # (ddsp.effects.Reverb(name="reverb", reverb_length=sample_rate, trainable=True), ["distortion2/signal"]),
            ],
            name="processor_group",
        )

        super(NeuroBass, self).__init__(
            sample_rate=sample_rate,
            example_secs=example_secs,
            embedding_loss=embedding_loss,
            decoder=decoder,
            processor_group=processor_group,
            name="neurobass",
        )
