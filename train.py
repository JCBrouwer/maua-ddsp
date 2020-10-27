import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
import absl

absl.logging.set_verbosity("info")
absl.logging.set_stderrthreshold("info")

import argparse
import ddsp
import models

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--save_dir", type=str, required=True)
parser.add_argument("--restore_dir", type=str, default="/tmp/ddsp")
parser.add_argument("--model", type=str, default="Default")
parser.add_argument("--embedding_loss", action="store_true")
parser.add_argument("--train_steps", type=int, default=300_000)
parser.add_argument("--steps_per_save", type=int, default=2500)
parser.add_argument("--steps_per_summary", type=int, default=250)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--sr", type=int, default=16000)
parser.add_argument("--example_secs", type=float, default=4)
parser.add_argument("--frame_rate", type=int, default=250)
args = parser.parse_args()

tfrecord = ddsp.training.data.TFRecordProvider(
    file_pattern=args.data_dir + "train.tfrecord*",
    example_secs=args.example_secs,
    sample_rate=args.sr,
    frame_rate=args.frame_rate,
)

strategy = ddsp.training.train_util.get_strategy()
with strategy.scope():
    model = models.__dict__[args.model](args.sr, args.example_secs, args.embedding_loss)
    trainer = ddsp.training.trainers.Trainer(
        model,
        strategy,
        learning_rate=1e-3,
        checkpoints_to_keep=10,
        grad_clip_norm=3.0,
        lr_decay_rate=0.98,
        lr_decay_steps=10000,
        restore_keys=None,
    )

ddsp.training.train_util.train(
    tfrecord,
    trainer,
    save_dir=args.save_dir,
    restore_dir=args.restore_dir,
    batch_size=args.batch_size,
    num_steps=args.train_steps,
    steps_per_save=args.steps_per_save,
    steps_per_summary=args.steps_per_summary,
    early_stop_loss_value=None,
    report_loss_to_hypertune=False,
)
