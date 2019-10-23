import tensorflow as tf
import os

def setup_gpu(gpu_id, verbose=True):
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)
    if verbose:
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.InteractiveSession(config=config)