import tensorflow as tf


def setup_gpu(gpu_id, verbose=True):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            gpu_ids = [int(s) for s in gpu_id.split(",")]
            gpus = [gpus[i] for i in gpu_ids]
            tf.config.experimental.set_visible_devices(gpus, "GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(
                "Found {} Physical GPUs, {} Logical GPU".format(
                    len(gpus), len(logical_gpus)
                )
            )
            print("Using GPU {}".format(gpu_ids))
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
