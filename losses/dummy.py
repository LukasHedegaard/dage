import tensorflow as tf
from utils.dataset_gen import DTYPE

def dummy_loss(
    y_true, 
    y_pred
):
    return tf.constant(0, dtype=DTYPE)