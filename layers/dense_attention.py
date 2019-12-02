import tensorflow as tf
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import math_ops
from tensorflow.keras.layers import Layer


class DenseAttention(Layer):

    def __init__(self,
                 classes,
                 batch_size,
                 inverse=False,
                 omit_intra_domain=True,
                 activation='softmax',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(DenseAttention, self).__init__(activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
        self.classes = int(classes)
        self.batch_size = int(batch_size)
        self.inverse = inverse
        self.omit_intra_domain = omit_intra_domain
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.supports_masking = True
        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point dtype %s' % (dtype,))
        input_shape = tensor_shape.TensorShape(input_shape) # N x D
        self.dims = tensor_shape.dimension_value(input_shape[-1])
        if self.dims is None:
            raise ValueError('The last dimension of the inputs to `DenseAttention` should be defined. Found `None`.')
        
        self.input_spec = InputSpec(min_ndim=2, axes={ -1: self.dims})
        weight_shape = [self.classes, self.dims, self.dims] # C x D x D
        bias_shape = [self.classes, self.dims] # C x D
        
        self.kernel = self.add_weight(
            'kernel',
            shape=weight_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True
        )
        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=bias_shape,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, labels):
        rank = len(inputs.shape)
        if rank != 2:
            raise ValueError('The input to `DenseAttention` should be of rank 2, but got input of shape %s'% inputs.shape)
        if K.is_sparse(inputs):
            raise ValueError("`DenseAttention` doesn't support SparseTensor as input")
            
        inputs = math_ops.cast(inputs, self._compute_dtype)
        # self.batch_size = tensor_shape.dimension_value(inputs.shape[0])
        
        # broadcast input to the number of classes
        inputs_shape_b = [self.classes, self.batch_size, self.dims]
        xTe = tf.broadcast_to(tf.expand_dims(inputs, axis=0), shape=inputs_shape_b)
        
        # apply attention-weights
        xB = tf.matmul(xTe, self.kernel)
        if self.use_bias:
            bias = tf.broadcast_to(tf.expand_dims(self.bias, axis=1), shape=inputs_shape_b)
            xB = tf.add(xB, bias)
        xBBx = tf.matmul(xB, xB, transpose_b=True)
        xBBx = tf.transpose(xBBx, perm=[1,2,0])
        
        # create masking according to samples classes
        labels_shape_b = [self.batch_size, self.batch_size, self.classes ]
        yTe = tf.broadcast_to(tf.expand_dims(labels, axis=1), shape=labels_shape_b)
        eTy = tf.broadcast_to(tf.expand_dims(labels, axis=0), shape=labels_shape_b)
        W = tf.equal(yTe, eTy) if not self.inverse else tf.not_equal(yTe, eTy)

        if self.omit_intra_domain:
            tile_size = [self.batch_size//2, self.batch_size//2, self.classes]
            zeros = tf.zeros(tile_size, dtype=tf.bool)
            ones = tf.ones(tile_size, dtype=tf.bool)
            mask = tf.concat([tf.concat([zeros, ones], axis=0),
                              tf.concat([ones, zeros], axis=0)], axis=1 )
            W = tf.logical_and(W, mask)

        A = tf.where(W, xBBx, tf.zeros_like(xBBx)) # mask using W
        
        if self.activation is not None:
            A = tf.reshape(A,[self.classes, self.batch_size*self.batch_size])
            A = self.activation(A) # pylint: disable=not-callable
            A = tf.reshape(A,[self.batch_size, self.batch_size, self.classes])
            A = tf.where(W, A, tf.zeros_like(A)) # mask again to remove spill-over from softmax
            
        outputs = tf.reduce_sum(A, axis=2) #reduce along the channel axis
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The innermost dimension of input_shape must be defined, but saw: %s'% input_shape)
        return [input_shape[0], input_shape[0]]

    def get_config(self):
        config = {
            'classes': self.classes,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(DenseAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))