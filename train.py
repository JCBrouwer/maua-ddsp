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
parser.add_argument("--gin_file", type=str, default=None)
parser.add_argument("--model", type=str, default="Default")
parser.add_argument("--embedding_loss", action="store_true")
parser.add_argument("--train_steps", type=int, default=300_000)
parser.add_argument("--steps_per_save", type=int, default=2500)
parser.add_argument("--steps_per_summary", type=int, default=250)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--sr", type=int, default=16000)
parser.add_argument("--example_secs", type=int, default=4)
parser.add_argument("--frame_rate", type=int, default=250)
args = parser.parse_args()

if args.model == "gin":
    import gin, pkg_resources

    # Add user folders to the gin search path.
    gin_path = pkg_resources.resource_filename(__name__, "gin")
    gin.add_config_file_search_path(gin_path)

    # Parse gin configs, later calls override earlier ones.
    with gin.unlock_config():
        gin.parse_config_file(os.path.join("optimization", "base.gin"))

        # Load operative_config if it exists (model has already trained).
        operative_config = train_util.get_latest_operative_config(restore_dir)
        if tf.io.gfile.exists(operative_config):
            logging.info("Using operative config: %s", operative_config)
            operative_config = cloud.make_file_paths_local(operative_config, gin_path)
            gin.parse_config_file(operative_config, skip_unknown=True)

        # User gin config and user hyperparameters from flags.
        if args.gin_file is not None:
            gin_file = cloud.make_file_paths_local(args.gin_file, gin_path)
        else:
            print("WARNING: --gin_file not supplied")

    strategy = ddsp.training.train_util.get_strategy()
    with strategy.scope():
        model = ddsp.models.get_model()
        trainer = trainers.Trainer(model, strategy)

    ddsp.training.train_util.train(
        data_provider=gin.REQUIRED, trainer=trainer, save_dir=save_dir, restore_dir=restore_dir
    )
else:
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
