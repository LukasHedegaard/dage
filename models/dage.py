import tensorflow as tf
M = tf.compat.v2.math
keras = tf.compat.v2.keras
K = keras.backend
from utils.dataset_gen import DTYPE
from math import ceil
from functools import reduce, partial
from enum import Enum

def Pair(name, embed_size, num_in_pair=2):
    ''' Collects the input layers into a pair. 
        Assummes that input dimensions of the inputs equal.
    '''
    layers = [
        tf.keras.layers.Concatenate(axis=1, name="{}_concat".format(name)),
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


class LossRelation(Enum):
    ALL = 1
    SOURCE_TARGET = 2
    SOURCE_TARGET_PAIR = 3

class LossFilter(Enum):
    ALL = 1
    KNN = 2
    EPSILON = 3


def take_all(W, xs, xt):
        return W


def relate_all(ys, yt, batch_size):
    N = 2*batch_size 
    y = tf.concat([ys, yt], axis = 0) 
    yTe = tf.broadcast_to(tf.expand_dims(y, axis=1), shape=(N, N))
    eTy = tf.broadcast_to(tf.expand_dims(y, axis=0), shape=(N, N))

    W = tf.equal(yTe, eTy)
    Wp = tf.not_equal(yTe, eTy)

    return W, Wp


def relate_source_target(ys, yt, batch_size):
    W_all, Wp_all = relate_all(ys, yt, batch_size)

    N = 2*batch_size 
    i = tf.constant([[False,True],[True,False]], dtype=tf.bool)
    for ax in range(2):
        i = keras.backend.repeat_elements(i, N//2, axis=ax)

    zeros = tf.zeros([N,N],dtype=tf.bool)
    W = tf.where(i, W_all, zeros)
    Wp = tf.where(i, Wp_all, zeros)

    return W, Wp


def relate_source_target_pair(ys, yt, batch_size):
    eq = tf.linalg.diag(tf.equal(ys, yt))
    neq = tf.linalg.diag(tf.not_equal(ys, yt))
    zeros = tf.zeros([batch_size, batch_size],dtype=tf.bool)
    W = tf.concat([tf.concat([zeros, eq], axis=0),
                   tf.concat([eq, zeros], axis=0)], axis=1)
    Wp = tf.concat([tf.concat([zeros, neq], axis=0),
                    tf.concat([neq, zeros], axis=0)], axis=1)

    return W, Wp


def make_embedding_loss(
    batch_size,
    relation_type: LossRelation, 
    filter_type: LossFilter, 
    param:int=None
): 
    relate = {
        LossRelation.ALL                : relate_all,
        LossRelation.SOURCE_TARGET      : relate_source_target,
        LossRelation.SOURCE_TARGET_PAIR : relate_source_target_pair,
    }[relation_type]

    filt = {
        LossFilter.ALL     : take_all,
        # LossFilter.KNN      : take_knn,
        # LossFilter.EPSILON  : take_epsilon,
    }[filter_type]

    def make_weights(xs, xt, ys, yt, batch_size):
        W, Wp = filt(
            W=relate(ys, yt, batch_size),
            xs=xs,
            xt=xt
        )
        return tf.cast(W, dtype=DTYPE), tf.cast(Wp, dtype=DTYPE)


    def loss_fn(y_true, y_pred):
        ''' Tensorflow implementation of our graph embedding loss
            Assumes the input layer to be linear, i.e. a dense layer without activation
        '''        
        ys = tf.argmax(tf.cast(y_true[:,0], dtype=tf.int32), axis=1)
        yt = tf.argmax(tf.cast(y_true[:,1], dtype=tf.int32), axis=1)
        xs = y_pred[:,0]
        xt = y_pred[:,1]
        θϕ = tf.transpose(tf.concat([xs,xt], axis=0))

        # NB: We would like to access the batch size dynamically, but currently, Keras doesn't seem to allow the command below (gives Dimension(None) )
        # batch_size = ys.get_shape()[0]

        # construct Weight matrix
        W, Wp = make_weights(xs, xt, ys, yt, batch_size)

        # construct Degree matrix
        D  = tf.linalg.diag(tf.reduce_sum(W,  axis=1)) 
        Dp = tf.linalg.diag(tf.reduce_sum(Wp, axis=1))

        # construct Graph Laplacian
        L  = tf.subtract(D, W)
        Lp = tf.subtract(Dp, Wp)
        
        # construct loss
        θϕLϕθ  = tf.matmul(θϕ, tf.matmul(L,  θϕ, transpose_b=True))
        θϕLpϕθ = tf.matmul(θϕ, tf.matmul(Lp, θϕ, transpose_b=True))

        loss = tf.linalg.trace(θϕLϕθ) / tf.linalg.trace(θϕLpϕθ)

        return loss

    return loss_fn
    


def model(
    model_base, 
    input_shape,
    output_shape,
    optimizer,
    batch_size,
    alpha=0.25,
    even_loss_weights=True,
    freeze_base=True,
    embed_size=128,
    dense_size=1024
):  
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

    model_base_output_shape = model_base.layers[-1].output_shape
    if type(model_base_output_shape) == list:
        model_base_output_shape = model_base_output_shape[0]
    if type(model_base_output_shape) == tuple:
        model_base_output_shape = model_base_output_shape[1:]

    model_mid = keras.Sequential([
        keras.layers.Input(shape=model_base_output_shape),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(dense_size, activation=None, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='dense'),
        keras.layers.BatchNormalization(momentum=0.9),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(embed_size, activation=None, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='embed'),
        keras.layers.BatchNormalization(momentum=0.9),
        # keras.layers.Activation('relu'),
    ], name='dense_layers')

    model_top = keras.Sequential([
        keras.layers.Input(shape=model_mid.layers[-1].output_shape[1:]),
        keras.layers.Activation('relu'), #from prev layer
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
        loss=loss(batch_size), 
        loss_weights=loss_weights(alpha, even_loss_weights), 
        optimizer=optimizer, 
        metrics={'preds':'accuracy', 'preds_1':'accuracy'},
    )

    return model

def loss(batch_size):
    return {
        'preds'  : keras.losses.categorical_crossentropy,
        'preds_1': keras.losses.categorical_crossentropy,
        'aux_out': make_embedding_loss(
            batch_size=batch_size,
            relation_type=LossRelation.ALL, 
            filter_type=LossFilter.ALL,
        )
    }

def loss_weights(alpha=0.25, even=1):
    return {
        'preds'  : 0.5*(1-alpha) if even else 0,
        'preds_1': 0.5*(1-alpha) if even else 1-alpha,
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
    # validation_steps = ceil(val_datasource_size/batch_size) if val_datasource_size else None
    # steps_per_epoch = ceil(datasource_size/batch_size)
    validation_steps = val_datasource_size//batch_size if val_datasource_size else None
    steps_per_epoch = datasource_size//batch_size

    model.fit( 
        datasource,
        validation_data=val_datasource,
        epochs=epochs, 
        steps_per_epoch=steps_per_epoch, 
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=verbose,
    )