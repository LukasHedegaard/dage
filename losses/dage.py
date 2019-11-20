import tensorflow as tf
K = tf.compat.v2.keras.backend
from utils.dataset_gen import DTYPE
from enum import Enum
from functools import partial

# Configuration parameters
class LossRelation(Enum):
    ALL = 1
    SOURCE_TARGET = 2
    SOURCE_TARGET_PAIR = 3

class LossFilter(Enum):
    ALL = 1
    KNN = 2
    EPSILON = 3

# Selectors
def take_all(W, xs, xt):
        return W

# Relations
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
        i = K.repeat_elements(i, N//2, axis=ax)

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

# Loss maker
def dage_loss(
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

        batch_size = tf.shape(ys)[0]

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


dage_full_loss = dage_loss(
    relation_type=LossRelation.ALL, 
    filter_type=LossFilter.ALL,
)