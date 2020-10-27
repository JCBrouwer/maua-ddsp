```
python distortion_test.py \
  --audio AUDIO \
  --sr SR
```

```
python prepare_data.py \
  --audio_dir AUDIO_DIR \
  --save_dir SAVE_DIR \
  --sr SR \
  --frame_rate FRAME_RATE \
  --example_secs EXAMPLE_SECS
```

```
python train.py \
  --data_dir DATA_DIR \
  --save_dir SAVE_DIR \
  --restore_dir RESTORE_DIR \
  --model MODEL \
  --embedding_loss # only if using 1 GPU \
  --train_steps TRAIN_STEPS \
  --steps_per_save STEPS_PER_SAVE \
  --steps_per_summary STEPS_PER_SUMMARY \
  --batch_size BATCH_SIZE \
  --sr SR \
  --example_secs EXAMPLE_SECS \
  --frame_rate FRAME_RATE
```

```
python timbre_transfer.py \
  --audio AUDIO \
  --ckpt CKPT \
  --sr SR \
  --no_adjust \
  --threshold THRESHOLD \
  --quiet QUIET \
  --autotune AUTOTUNE \
  --pitch_shift PITCH_SHIFT \
  --loudness_shift LOUDNESS_SHIFT
```
