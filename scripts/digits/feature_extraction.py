import subprocess
from datetime import datetime
from pathlib import Path

DESCRIPTION = "Extract features from pretrauned models."
GPU_ID = "0"
EPOCHS = "50"

SOURCE = "usps"
TARGET = "mnist"
SEED = "2"

# ====== Tune source =======
weights_path = "runs/tune_source/tune_source_digits_usps_mnist/uspsmnist_2_20210503214324/checkpoints/cp-best.ckpt"

method = "tune_source"
experiment_id = f"{method}_digits_features_{SOURCE}_{TARGET}"
dir_name = Path("runs") / method / experiment_id
dir_name.mkdir(parents=True, exist_ok=True)

# fmt: off
subprocess.call([
    "python3","run.py",
    "--source",             SOURCE,
    "--target",             TARGET,
    "--seed",               SEED,
    "--method",             method,
    "--architecture",       "single_stream",
    "--training_regimen",   "regular",
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
    # "--num_source_samples_per_class",  "700",
    # "--num_target_samples_per_class",  "10",
    "--mode",               "features",
])
# fmt: on


# ====== Tune target =======
weights_path = "runs/tune_target/tune_target_digits_usps_mnist/uspsmnist_2_20210503214324/checkpoints/cp-best.ckpt"
method = "tune_target"
experiment_id = f"{method}_digits_features_{SOURCE}_{TARGET}"
dir_name = Path("runs") / method / experiment_id
dir_name.mkdir(parents=True, exist_ok=True)

# fmt: off
subprocess.call([
    "python3","run.py",
    "--source",             SOURCE,
    "--target",             TARGET,
    "--seed",               SEED,
    "--method",             method,
    "--architecture",       "single_stream",
    "--training_regimen",   "regular",
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
    # "--num_source_samples_per_class",  "700",
    # "--num_target_samples_per_class",  "10",
    "--mode",               "features",
])
# fmt: on

# ====== DAGE-LDA =======
weights_path = "runs/dage/dage_digits_usps_mnist/uspsmnist_2_20210413122944/checkpoints/cp-best.ckpt"

method = "dage"
experiment_id = f"{method}_digits_features_{SOURCE}_{TARGET}"
dir_name = Path("runs") / method / experiment_id
dir_name.mkdir(parents=True, exist_ok=True)

# fmt: off
subprocess.call([
    "python3","run.py",
    "--source",                             SOURCE,
    "--target",                             TARGET,
    "--seed",                               SEED,
    "--method",                             method,
    "--architecture",                       "two_stream_pair_embeds",
    "--training_regimen",                   "regular",
    "--model_base",                         "conv2",
    "--features",                           "images",
    "--epochs",                             EPOCHS,
    "--batch_size",                         "128",
    "--training_regimen",                   "regular",
    "--num_source_samples_per_class",       "700",
    "--num_target_samples_per_class",       "10",
    "--connection_type",                    "SOURCE_TARGET",
    "--weight_type",                        "INDICATOR",
    "--connection_filter_type",             "ALL",
    "--penalty_connection_filter_type",     "ALL",
    "--batch_norm",                         "1",
    "--optimizer",                          "adam",
    "--learning_rate",                      "0.0035753161317240803",
    "--learning_rate_decay",                "1.572832625907872e-05",
    "--dropout",                            "0.2412034416347774",
    "--l2",                                 "0.0003828726839315707",
    "--momentum",                           "0.984569",
    "--loss_alpha",                         "0.47586281871846964",
    "--loss_weights_even",                  "0.5632755719763838",
    "--ratio",                              "3",
    "--resize_mode",                        "2",
    "--mode",                               "features",
    "--from_weights",                       weights_path,
])
# fmt: on
