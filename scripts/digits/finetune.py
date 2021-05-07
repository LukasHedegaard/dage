import subprocess
from datetime import datetime
from pathlib import Path

DESCRIPTION = "Train on source data, then tune on target data."
GPU_ID = "0"
EPOCHS = "50"


for seed in [1, 2, 3, 4, 5]:
    seed = str(seed)
    for source, target, num_source, num_target in [
        ("mnist", "mnist_m", 5000, 10),
        ("mnist", "usps", 5000, 10),
        ("usps", "mnist", 700, 10),
        ("mnist", "svhn", 5000, 10),
        ("svhn", "mnist", 700, 10),
    ]:
        # Train on source, test on target
        method = "tune_source"
        experiment_id = f"{method}_digits_{source}_{target}"
        dir_name = Path("runs") / method / experiment_id
        dir_name.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # fmt: off
        subprocess.call([
            "python3","run.py",
            "--source",             source,
            "--target",             target,
            "--seed",               seed,
            "--method",             method,
            "--architecture",       "single_stream",
            "--training_regimen",   "regular",
            "--timestamp",          timestamp,
            "--learning_rate",      "1e-5",
            "--epochs",             EPOCHS,
            "--gpu_id",             GPU_ID,
            "--optimizer",          "adam",
            "--experiment_id",      experiment_id,
            "--model_base",         "conv2",
            "--features",           "images",
            "--batch_size",         "12",
            "--augment",            "1",
            "--resize_mode",        "2",
            "--num_source_samples_per_class",  str(num_source),
            "--num_target_samples_per_class",  str(num_target),
        ])
        # fmt: on

        weights_path = str(
            Path("runs")
            / method
            / experiment_id
            / f"{source}{target}_{seed}_{timestamp}"
            / "checkpoints"
            / "cp-best.ckpt"
        )

        # Finetune on target
        method = "tune_target"
        experiment_id = f"{method}_digits_{source}_{target}"
        dir_name = Path("runs") / method / experiment_id
        dir_name.mkdir(parents=True, exist_ok=True)

        # fmt: off
        subprocess.call([
            "python3","run.py",
            "--source",             source,
            "--target",             target,
            "--seed",               seed,
            "--method",             method,
            "--architecture",       "single_stream",
            "--training_regimen",   "regular",
            "--timestamp",          timestamp,
            "--learning_rate",      "1e-5",
            "--epochs",             EPOCHS,
            "--gpu_id",             GPU_ID,
            "--optimizer",          "adam",
            "--experiment_id",      experiment_id,
            "--model_base",         "conv2",
            "--features",           "images",
            "--batch_size",         "12",
            "--augment",            "1",
            "--resize_mode",        "2",
            "--from_weights",       weights_path,
            "--num_source_samples_per_class",  str(num_source),
            "--num_target_samples_per_class",  str(num_target),
        ])
        # fmt: on
