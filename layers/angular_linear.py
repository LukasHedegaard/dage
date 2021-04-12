import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.framework import dtypes, tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints, initializers, regularizers
from tensorflow.python.keras.engine.input_spec import InputSpec

from layers.l2norm import l2norm_instance


class AngularLinear(Layer):
    """This layer was used in the source code for d-SNE
    I was not able to find any source describing it.
    """

    def __init__(
        self,
        units,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        **kwargs
    ):
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("input_dim"),)

        super(AngularLinear, self).__init__(**kwargs)
        self.units = int(units)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "Unable to build `AngularLinear` layer with non-floating point dtype %s"
                % (dtype,)
            )
        input_shape = tensor_shape.TensorShape(input_shape)  # N x D
        self.dims = tensor_shape.dimension_value(input_shape[-1])
        if self.dims is None:
            raise ValueError(
                "The last dimension of the inputs to `AngularLinear` should be defined. Found `None`."
            )
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.dims})
        self.kernel = self.add_weight(
            "kernel",
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        self.built = True

    def call(self, inputs):
        rank = len(inputs.shape)
        if rank != 2:
            raise ValueError(
                "The input to `AngularLinear` should be of rank 2, but got input of shape %s"
                % inputs.shape
            )
        if K.is_sparse(inputs):
            raise ValueError("`AngularLinear` doesn't support SparseTensor as input")

        x_norm = l2norm_instance(inputs)
        w_norm = l2norm_instance(self.kernel)
        dot = tf.matmul(x_norm, w_norm)
        outputs = tf.clip_by_value(dot, -1, 1)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                "The innermost dimension of input_shape must be defined, but saw: %s"
                % input_shape
            )
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = {
            "units": self.units,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
        }
        base_config = super(AngularLinear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
