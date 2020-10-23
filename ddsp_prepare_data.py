import glob
import os

AUDIO_DIR = "/home/hans/datasets/neuro-bass-ddsp"
AUDIO_FILEPATTERN = AUDIO_DIR + "/16bit/*"
SAVE_DIR = "/home/hans/modelzoo/neuro-bass-ddsp/"
os.makedirs(SAVE_DIR, exist_ok=True)
TRAIN_TFRECORD = os.path.join(AUDIO_DIR, "16bit/train.tfrecord")
TRAIN_TFRECORD_FILEPATTERN = TRAIN_TFRECORD + "*"
data_dir = os.path.join(AUDIO_DIR, "16bit")
dataset_files = glob.glob(data_dir + "/*")

# Make a new dataset.
if not glob.glob(AUDIO_FILEPATTERN):
    raise ValueError("No audio files found. Please use the previous cell to upload.")

if not glob.glob(TRAIN_TFRECORD_FILEPATTERN):
    from ddsp.training.data_preparation import prepare_tfrecord_lib

    prepare_tfrecord_lib.prepare_tfrecord(
        dataset_files,
        TRAIN_TFRECORD,
        num_shards=16,
        sample_rate=16000,  # 44100,
        frame_rate=250,  # 630,
        window_secs=2,
        hop_secs=1,
        pipeline_options="",
    )

    from ddsp.colab import colab_utils
    import ddsp.training

    data_provider = ddsp.training.data.TFRecordProvider(TRAIN_TFRECORD_FILEPATTERN)
    dataset = data_provider.get_dataset(shuffle=False)
    PICKLE_FILE_PATH = os.path.join(SAVE_DIR, "dataset_statistics.pkl")

    colab_utils.save_dataset_statistics(data_provider, PICKLE_FILE_PATH)

from ddsp.colab import colab_utils
import ddsp.training
from matplotlib import pyplot as plt
import numpy as np

data_provider = ddsp.training.data.TFRecordProvider(
    TRAIN_TFRECORD_FILEPATTERN  # , example_secs=2, sample_rate=44100, frame_rate=630
)
dataset = data_provider.get_dataset(shuffle=False)

try:
    ex = next(iter(dataset))
    print(ex)
except StopIteration:
    raise ValueError(
        "TFRecord contains no examples. Please try re-running the pipeline with " "different audio file(s)."
    )

# import librosa.plot

# librosa.specplot(ex["audio"])
# librosa.play(ex["audio"])

# f, ax = plt.subplots(3, 1, figsize=(14, 4))
# x = np.linspace(0, 4.0, 1000)
# ax[0].set_ylabel("loudness_db")
# ax[0].plot(x, ex["loudness_db"])
# ax[1].set_ylabel("F0_Hz")
# ax[1].set_xlabel("seconds")
# ax[1].plot(x, ex["f0_hz"])
# ax[2].set_ylabel("F0_confidence")
# ax[2].set_xlabel("seconds")
# ax[2].plot(x, ex["f0_confidence"])
# plt.show()

# %reload_ext tensorboard
# import tensorboard as tb
# tb.notebook.start('--logdir "{}"'.format(SAVE_DIR))

python -m ddsp.training.ddsp_run  \
  --mode=train \
  --alsologtostderr \
  --save_dir="/home/hans/modelzoo/neuro-bass-ddsp/" \
  --gin_file=models/solo_instrument.gin \
  --gin_file=datasets/tfrecord.gin \
  --gin_param="TFRecordProvider.file_pattern='/home/hans/datasets/neuro-bass-ddsp/16bit/train.tfrecord*'" \
  --gin_param="batch_size=16" \
  --gin_param="train_util.train.num_steps=300000" \
  --gin_param="train_util.train.steps_per_save=3000" \
  --gin_param="trainers.Trainer.checkpoints_to_keep=10"

# from ddsp.colab.colab_utils import play, specplot
# import ddsp.training
# import gin
# from matplotlib import pyplot as plt
# import numpy as np

# data_provider = ddsp.training.data.TFRecordProvider(TRAIN_TFRECORD_FILEPATTERN)
# dataset = data_provider.get_batch(batch_size=1, shuffle=False)

# try:
#   batch = next(iter(dataset))
# except OutOfRangeError:
#   raise ValueError(
#       'TFRecord contains no examples. Please try re-running the pipeline with '
#       'different audio file(s).')

# # Parse the gin config.
# gin_file = os.path.join(SAVE_DIR, 'operative_config-0.gin')
# gin.parse_config_file(gin_file)

# # Load model
# model = ddsp.training.models.Autoencoder()
# model.restore(SAVE_DIR)

# # Resynthesize audio.
# audio_gen = model(batch, training=False)
# audio = batch['audio']

# print('Original Audio')
# specplot(audio)
# play(audio)

# print('Resynthesis')
# specplot(audio_gen)
# play(audio_gen)
