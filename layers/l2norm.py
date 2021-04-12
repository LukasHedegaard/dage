import tensorflow as tf
from tensorflow.keras.layers import Layer


def l2norm_instance(inputs):
    axis = tuple(range(1, len(inputs.shape)))
    eps = tf.constant(1e-12)
    return tf.transpose(
        tf.transpose(inputs) / tf.sqrt(tf.reduce_sum(inputs ** 2, axis=axis) + eps)
    )


class L2NormInstance(Layer):
    def __init__(self):
        super(L2NormInstance, self).__init__()
        self.eps = tf.Variable(1e-12)

    def build(self, input_shape):
        self.axis = tuple(range(1, len(input_shape)))
        self.in_shape = input_shape

    def call(self, inputs):
        return tf.transpose(
            tf.transpose(inputs)
            / tf.sqrt(tf.reduce_sum(inputs ** 2, axis=self.axis) + self.eps)
        )

    def compute_output_shape(self, input_shape):
        return input_shape
