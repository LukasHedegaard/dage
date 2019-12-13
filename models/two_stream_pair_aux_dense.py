import tensorflow as tf
keras = tf.compat.v2.keras
from utils.dataset_gen import DTYPE
from functools import reduce
from layers import Pair
from losses import dummy_loss
from models.common import freeze, get_output_shape, model_dense, model_preds
from math import ceil


def model(
    model_base, 
    input_shape,
    output_shape,
    optimizer,
    batch_size,
    aux_loss=dummy_loss,
    loss_alpha=0.25,
    loss_weights_even=True,
    num_unfrozen_base_layers=0,
    embed_size=128,
    dense_size=1024,
    aux_dense_size=31,
    l2 = 0.0001,
    batch_norm=True
):
    in1 = keras.layers.Input(shape=input_shape, name='input_source')
    in2 = keras.layers.Input(shape=input_shape, name='input_target')

    freeze(model_base, num_leave_unfrozen=num_unfrozen_base_layers)

    model_mid = model_dense(input_shape=get_output_shape(model_base), dense_size=dense_size, embed_size=embed_size, l2=l2, batch_norm=batch_norm)
    model_top = model_preds(input_shape=get_output_shape(model_mid), output_shape=output_shape, l2=l2)

    # weight sharing is used: the same instance of model_base, and model_mid is used for both streams
    mid1 = model_mid(model_base(in1))
    mid2 = model_mid(model_base(in2))
    
    aux_dense = keras.layers.Dense(aux_dense_size, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='aux_dense')
    aux_out = Pair(name='aux_out', embed_size=aux_dense_size)([aux_dense(mid1), aux_dense(mid2)])

    # the original authors had only a single prediction output, and had to feed every batch twice, flipping source and target on the second run
    # we instead create two prediction layers (shared weights) as a performance optimisation (base and mid only run once)
    out1 = model_top(mid1)
    out2 = model_top(mid2)

    model = keras.models.Model(inputs=[in1, in2], outputs=[out1, out2, aux_out])
    model_test = keras.models.Model(inputs=[in1], outputs=[out1])

    loss = {
        'preds'  : keras.losses.categorical_crossentropy,
        'preds_1': keras.losses.categorical_crossentropy,
        'aux_out': aux_loss
    }

    loss_weights = {
        'preds'  : 0.5*(1-loss_alpha) if loss_weights_even else 0,
        'preds_1': 0.5*(1-loss_alpha) if loss_weights_even else 1-loss_alpha,
        'aux_out': loss_alpha
    }
    
    model.compile(
        loss=loss, 
        loss_weights=loss_weights, 
        optimizer=optimizer, 
        metrics={'preds':'accuracy', 'preds_1':'accuracy'},
    )

    return model, model_test
