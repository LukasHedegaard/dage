import tensorflow as tf
keras = tf.compat.v2.keras
from utils.dataset_gen import DTYPE
from functools import reduce
from layers import Pair, DenseAttention
from losses import dummy_loss
from models.common import freeze, get_output_shape, model_dense, model_preds, model_attention
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
    freeze_base=True,
    embed_size=128,
    dense_size=1024,
    attention_embed_size=1024
):
    in_src = keras.layers.Input(shape=input_shape, name='input_source')
    in_tgt = keras.layers.Input(shape=input_shape, name='input_target')
    # labels need to be passed as an input in order to let the DAGE loss access them
    lbl_src = keras.layers.Input(shape=output_shape, name='label_source') 
    lbl_tgt = keras.layers.Input(shape=output_shape, name='label_target') 

    model_base = model_base
    if freeze_base:
        freeze(model_base)
    else:
        freeze(model_base, num_leave_unfrozen=4)

    model_mid   = model_dense(input_shape=get_output_shape(model_base), dense_size=dense_size, embed_size=embed_size)
    model_top   = model_preds(input_shape=get_output_shape(model_mid), output_shape=output_shape)

    # weight sharing is used: the same instance of model_base, and model_mid is used for both streams
    base_out_src = model_base(in_src)
    base_out_tgt = model_base(in_tgt)

    mid_out_src = model_mid(base_out_src)
    mid_out_tgt = model_mid(base_out_tgt)

    preds_src = model_top(mid_out_src)
    preds_tgt = model_top(mid_out_tgt)


    # Setup for DAGE loss
    # pair_out = Pair(name='aux_out', embed_size=embed_size)([mid_out_src, mid_out_tgt])
    concat_out = keras.layers.Concatenate(axis=0)([mid_out_src, mid_out_tgt])
    concat_lbl = keras.layers.Concatenate(axis=0)([lbl_src, lbl_src])

    att_out    = DenseAttention(classes=output_shape, input_shape=get_output_shape(model_mid))(concat_out, concat_lbl)
    att_p_out  = DenseAttention(classes=output_shape, input_shape=get_output_shape(model_mid))(concat_out, concat_lbl)

    model = keras.models.Model(inputs=[in_src, in_tgt, lbl_src, lbl_tgt], outputs=[preds_src, preds_tgt, att_out, att_p_out])
    model_test = keras.models.Model(inputs=[in_tgt], outputs=[preds_tgt])

    # Add losses 
    dage_loss = aux_loss(lbl_src, lbl_tgt, mid_out_src, mid_out_tgt, att_out, att_p_out)
    ce_loss_src = tf.reduce_mean(keras.losses.categorical_crossentropy(lbl_src, preds_src))
    ce_loss_tgt = tf.reduce_mean(keras.losses.categorical_crossentropy(lbl_tgt, preds_tgt))

    ce_loss_weight_src = tf.cast(0.5*(1-loss_alpha) if loss_weights_even else 0, dtype=DTYPE)
    ce_loss_weight_tgt = tf.cast(0.5*(1-loss_alpha) if loss_weights_even else 1-loss_alpha, dtype=DTYPE)
    dage_loss_weight   = tf.cast(loss_alpha, dtype=DTYPE)

    model.add_loss(tf.scalar_mul(dage_loss_weight,   dage_loss  ))
    model.add_loss(tf.scalar_mul(ce_loss_weight_src, ce_loss_src))
    model.add_loss(tf.scalar_mul(ce_loss_weight_tgt, ce_loss_tgt))

    # Add metrics
    model.add_metric(ce_loss_src, name='ce_loss_src', aggregation='mean')
    model.add_metric(ce_loss_tgt, name='ce_loss_tgt', aggregation='mean')
    model.add_metric(dage_loss, name='aux_loss', aggregation='mean')
    model.add_metric(tf.keras.metrics.categorical_accuracy(lbl_src, preds_src), name='preds_acc_src', aggregation='mean')
    model.add_metric(tf.keras.metrics.categorical_accuracy(lbl_tgt, preds_tgt), name='preds_acc', aggregation='mean')
    # model.add_metric(tf.keras.layers.Flatten()(att_out), name='attention')
    
    # Compile
    model.compile(optimizer=optimizer)

    return model, model_test

def train(
    model, 
    datasource, 
    datasource_size, 
    epochs, 
    batch_size, 
    callbacks, 
    verbose=1, 
    val_datasource=None, 
    val_datasource_size=None 
):
    validation_steps = val_datasource_size//batch_size if val_datasource_size else None
    steps_per_epoch = datasource_size//batch_size

    if not val_datasource_size:
        val_datasource = None
        validation_steps = None

    model.fit( 
        datasource,
        validation_data=val_datasource,
        epochs=epochs, 
        steps_per_epoch=steps_per_epoch, 
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=verbose,
    )
