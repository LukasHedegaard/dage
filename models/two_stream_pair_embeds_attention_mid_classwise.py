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
    num_unfrozen_base_layers=0,
    embed_size=128,
    dense_size=1024,
    attention_activation='softmax',
    l2 = 0.0001,
    batch_norm=True,
):
    in_src = keras.layers.Input(shape=input_shape, name='input_source')
    in_tgt = keras.layers.Input(shape=input_shape, name='input_target')
    # labels need to be passed as an input in order to let the DAGE loss access them
    lbl_src = keras.layers.Input(shape=output_shape, name='label_source') 
    lbl_tgt = keras.layers.Input(shape=output_shape, name='label_target') 

    model_base = model_base
    freeze(model_base, num_leave_unfrozen=num_unfrozen_base_layers)

    model_mid   = model_dense(input_shape=get_output_shape(model_base), dense_size=dense_size, embed_size=embed_size, l2=l2, batch_norm=batch_norm)
    model_top   = model_preds(input_shape=get_output_shape(model_mid), output_shape=output_shape, l2=l2)

    # weight sharing is used: the same instance of model_base, and model_mid is used for both streams
    base_out_src = model_base(in_src)
    base_out_tgt = model_base(in_tgt)

    mid_out_src = model_mid(base_out_src)
    mid_out_tgt = model_mid(base_out_tgt)

    preds_src = model_top(mid_out_src)
    preds_tgt = model_top(mid_out_tgt)


    # Setup for DAGE loss
    concat_out = keras.layers.Concatenate(axis=0)([mid_out_src, mid_out_tgt])
    concat_lbl = keras.layers.Concatenate(axis=0)([lbl_src, lbl_src])

    concat_out = keras.layers.Lambda(lambda x: keras.backend.stop_gradient(x))(concat_out)
    concat_lbl = keras.layers.Lambda(lambda x: keras.backend.stop_gradient(x))(concat_lbl)

    att_out = DenseAttention(
        input_shape=get_output_shape(model_mid), 
        classes=output_shape, 
        activation=attention_activation, 
        batch_size=2*batch_size
    )(concat_out, concat_lbl)

    att_p_out = DenseAttention(
        input_shape=get_output_shape(model_mid), 
        classes=output_shape, 
        activation=attention_activation, 
        batch_size=2*batch_size,
        inverse=True
    )(concat_out, concat_lbl)

    model = keras.models.Model(inputs=[in_src, in_tgt, lbl_src, lbl_tgt], outputs=[mid_out_src, mid_out_tgt, preds_src, preds_tgt, att_out, att_p_out])
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

def train_flipping(
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

    train_iter = iter(datasource)
    
    for e in range(1,epochs+1):
        if verbose:
            print('Epoch {}/{}.'.format(e, epochs))

        for step in range(steps_per_epoch):
            batch = next(train_iter)

            source_loss = model.train_on_batch(batch)

            target_loss = model.train_on_batch({
                'input_source': batch['input_target'], 
                'input_target': batch['input_source'],
                'label_source': batch['label_target'],
                'label_target': batch['label_source'],
            })

            if step % 5 == 0 and verbose:
                print(' Step {}/{}'.format(step, steps_per_epoch))
                print('  Source Pass:  {}'.format(
                    '  '.join(['{} {:0.4f}'.format(model.metrics_names[i], source_loss[i]) for i in range(len(source_loss))])
                ))
                print('  Target Pass:  {}'.format(
                    '  '.join(['{} {:0.4f}'.format(model.metrics_names[i], target_loss[i]) for i in range(len(target_loss))])
                ))

        val_iter = iter(val_datasource)
        val_loss = []
        for step in range(validation_steps):
            batch = next(val_iter)
            val_loss.append(model.test_on_batch(batch))

        val_loss_avg = reduce(
            lambda n, o: [n[i]+o[i] for i in range(len(val_loss))],
            val_loss, 
            [0 for _ in val_loss[0]]
        )

        if verbose:
            print('  Validation:  {}'.format(
                '  '.join(['{} {:0.4f}'.format(model.metrics_names[i], val_loss_avg[i]) for i in range(len(val_loss_avg))])
            ))