import abc
import tensorflow as tf
from typing import List


def _create_weights(name: str, shape: List[int], he_init=False, trainable=True):
    if he_init:
        # he_normal corresponds to tf.variance_scaling_initializer(scale=2.0, mode='fan_in')
        init = tf.compat.v2.initializers.he_normal()
    else:
        # Glorot and Xavier are same
        init = tf.compat.v2.initializers.glorot_uniform()
    return tf.Variable(initial_value=init(shape), name=name, trainable=trainable)


def _create_bias(name: str, shape: List[int]):
    bias = tf.Variable(
        name=name,
        initial_value=tf.compat.v2.constant_initializer(0.01)(shape)
    )
    return bias


class ModelBase(tf.compat.v2.Module):
    def __init__(self, name):
        super().__init__(name)

    def compute_loss(self, y_true, logits):
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=logits)
        loss = tf.reduce_mean(losses, name='loss')
        return loss

    @abc.abstractmethod
    def call(self, inputs: dict, training=False):
        raise NotImplementedError('Must be implemented in subclasses.')


class DenseLayer(tf.compat.v2.Module):
    def __init__(self, 
        input_dim: int, 
        output_dim: int, 
        dropout_rate: float = None,
        activation: str = 'relu', 
        name: str = 'dense'
    ):
        super().__init__(name)
        self.dropout_rate = dropout_rate
        self.activation = activation
        if activation is not None and not self.activation == 'relu':
            raise ValueError('Only ReLU activation function is supported')
        with self.name_scope:
            self.weights = _create_weights(name='weights', shape=[input_dim, output_dim])
            self.bias = _create_bias(name='bias', shape=[output_dim])

    @tf.Module.with_name_scope
    def call(self, input_tensor: tf.Tensor, training=False):
        x = tf.matmul(input_tensor, self.weights)
        x = tf.nn.bias_add(x, self.bias)
        if training is True and self.dropout_rate is not None:
            x = tf.nn.dropout(x, rate=self.dropout_rate)
        if self.activation == 'relu':
            x = tf.nn.relu(x)
        return x
