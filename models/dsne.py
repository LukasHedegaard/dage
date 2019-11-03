import tensorflow as tf
M = tf.compat.v2.math
keras = tf.compat.v2.keras
K = keras.backend
from utils.dataset_gen import DTYPE
from math import ceil
from functools import reduce

def Pair(name, embed_size, num_in_pair=2):
    ''' Collects the input layers into a pair. 
        Assummes that input dimensions of the inputs equal.
    '''
    layers = [
        tf.keras.layers.Concatenate(axis=1),
        tf.keras.layers.Reshape(
            (num_in_pair, embed_size), 
            input_shape=(num_in_pair*embed_size,), 
            name=name
        )
    ]
    def call(x):
        for layer in layers:
            x = layer(x)
        return x
    return call

def dnse_loss(
    batch_size = 16,
    embed_size = 128,
    margin=1
):
    batch_size_target = batch_size_source = batch_size

    def loss(y_true, y_pred):
        ''' Tensorflow implementation of d-SNE loss. 
            Original Mxnet implementation found at https://github.com/aws-samples/d-SNE.
            @param y_true: tuple or array of two elements, containing source and target features
            @param y_pred: tuple or array of two elements, containing source and taget labels
        '''

        xs, xt = y_pred[:,0], y_pred[:,1]
        ys = tf.argmax(tf.cast(y_true[:,0], dtype=tf.int32), axis=1)
        yt = tf.argmax(tf.cast(y_true[:,1], dtype=tf.int32), axis=1)

        # The original implementation provided an optional feature-normalisation (L2) here. We'll skip it

        xs_rpt = tf.broadcast_to(tf.expand_dims(xs, axis=0), shape=(batch_size_target, batch_size_source, embed_size))
        xt_rpt = tf.broadcast_to(tf.expand_dims(xt, axis=1), shape=(batch_size_target, batch_size_source, embed_size))

        dists = tf.reduce_sum(tf.square(xt_rpt - xs_rpt), axis=2)

        yt_rpt = tf.broadcast_to(tf.expand_dims(yt, axis=1), shape=(batch_size_target, batch_size_source))
        ys_rpt = tf.broadcast_to(tf.expand_dims(ys, axis=0), shape=(batch_size_target, batch_size_source))

        y_same = tf.equal(yt_rpt, ys_rpt)
        y_diff = tf.not_equal(yt_rpt, ys_rpt)

        intra_cls_dists = tf.multiply(dists, tf.cast(y_same, dtype=DTYPE))
        inter_cls_dists = tf.multiply(dists, tf.cast(y_diff, dtype=DTYPE))

        max_dists = tf.reduce_max(dists, axis=1, keepdims=True)
        max_dists = tf.broadcast_to(max_dists, shape=(batch_size_target, batch_size_source))
        revised_inter_cls_dists = tf.where(y_same, max_dists, inter_cls_dists)

        max_intra_cls_dist = tf.reduce_max(intra_cls_dists, axis=1)
        min_inter_cls_dist = tf.reduce_min(revised_inter_cls_dists, axis=1)

        loss = tf.nn.relu(max_intra_cls_dist - min_inter_cls_dist + margin)

        return loss

    return loss


def model(
    model_base, 
    input_shape,
    output_shape,
    optimizer,
    batch_size,
    alpha=0.25,
    even_loss_weights=True,
    freeze_base=True,
):
    # ds_iter = dataset.make_one_shot_iterator()
    # x1, x2, y1, y2, y12 = ds_iter.get_next()
    # ds_iter = iter(dataset)
    # x1, x2, y1, y2, y12 = next(ds_iter)

    # in1 = keras.layers.Input(tensor=x1, name='input_source')
    # in2 = keras.layers.Input(tensor=x2, name='input_target')

    embed_size = 128
    
    in1 = keras.layers.Input(shape=input_shape, name='input_source')
    in2 = keras.layers.Input(shape=input_shape, name='input_target')

    model_base = model_base
    if freeze_base:
        for layer in model_base.layers:
            layer.trainable = False
    else:
        num_unfreeze_base = 4
        for layer in model_base.layers[:-num_unfreeze_base]:
            layer.trainable = False

    model_mid = keras.Sequential([
        keras.layers.Input(shape=model_base.layers[-1].output_shape[1:]),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation=None, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='dense'),
        keras.layers.BatchNormalization(momentum=0.9),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation=None, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='embed'),
        keras.layers.BatchNormalization(momentum=0.9),
        keras.layers.Activation('relu'),
    ], name='dense_layers')

    model_top = keras.Sequential([
        keras.layers.Input(shape=model_mid.layers[-1].output_shape[1:]),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(output_shape, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='logits'),
        keras.layers.Activation('softmax', name='preds'),
    ], name='preds')

    # weight sharing is used: the same instance of model_base, and model_mid is used for both streams
    mid1 = model_mid(model_base(in1))
    mid2 = model_mid(model_base(in2))

    # the original authors had only a single prediction output, and had to feed every batch twice, flipping source and target on the second run
    # we instead create two prediction layers (shared weights) as a performance optimisation (base and mid only run once)
    out1 = model_top(mid1)
    out2 = model_top(mid2)

    aux_out = Pair(name='aux_out', embed_size=embed_size)([mid1, mid2])

    model = keras.models.Model(
        inputs=[in1, in2],
        outputs=[out1, out2, aux_out]
    )
    
    model.compile(
        loss=loss(batch_size=batch_size, embed_size=embed_size, margin=1), 
        loss_weights=loss_weights(alpha, even_loss_weights), 
        optimizer=optimizer, 
        metrics=['accuracy'],
    )

    return model

def loss(batch_size=16, embed_size=128, margin=1):
    return {
        'preds'  : keras.losses.categorical_crossentropy,
        'preds_1': keras.losses.categorical_crossentropy,
        'aux_out': dnse_loss(batch_size=16, embed_size=128, margin=1)
    }

def loss_weights(alpha=0.25, even=True):
    return {
        'preds'  : 0.5*(1-alpha) if even else 1-alpha,
        'preds_1': 0.5*(1-alpha) if even else 0,
        'aux_out': alpha
    }

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
    validation_steps = ceil(val_datasource_size/batch_size) if val_datasource_size else None
    steps_per_epoch = ceil(datasource_size/batch_size)

    model.fit( 
        datasource,
        validation_data=val_datasource,
        epochs=epochs, 
        steps_per_epoch=steps_per_epoch, 
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=verbose,
    )