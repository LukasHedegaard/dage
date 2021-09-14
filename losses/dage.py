from enum import Enum
from functools import partial

import tensorflow as tf

from utils.dataset_gen import DTYPE

K = tf.compat.v2.keras.backend


class ConnectionType(Enum):
    ALL = "ALL"
    SOURCE_TARGET = "SOURCE_TARGET"
    SOURCE_TARGET_PAIR = "SOURCE_TARGET_PAIR"
    ST_INT_ALL_PEN = "ST_INT_ALL_PEN"


class FilterType(Enum):
    ALL = "ALL"
    KNN = "KNN"  # k-nearest-neighbours
    KFN = "KFN"  # k-farthest-neighbours
    KSD = "KSD"  # k-smallest-distances
    EPSILON = "EPSILON"  # epsilon margin


class WeightType(Enum):
    INDICATOR = "INDICATOR"
    GAUSSIAN = "GAUSSIAN"


def string2weight_type(s):
    return


# Util
def counter():
    c = 0
    while True:
        c += 1
        yield c


zeros_count = counter()
ones_count = counter()


# We need to wrap the zeros and ones function because unique names are mandatory in the graph
def zeros(shape, dtype=tf.bool):
    name = "zeros_{}".format(next(zeros_count))
    return tf.zeros(shape, dtype, name)


def ones(shape, dtype=tf.bool):
    name = "ones_{}".format(next(ones_count))
    return tf.ones(shape, dtype, name)


def str2enum(s, E):
    if isinstance(s, str):
        return E[s.upper()]
    elif isinstance(s, E):
        return s
    else:
        raise ValueError("{} should be either be {} or {}".format(s, str, E))


def get_filt_dists(W_Wp, xs, xt):
    # Seing as the distances are also calculated within the fisher ratio of the embedding loss, there might be a performance optimisation to be made here
    W, Wp = W_Wp

    batch_size, embed_size = tf.shape(xs)[0], tf.shape(xs)[1]
    N = 2 * batch_size
    x = tf.concat([xs, xt], axis=0)

    xTe = tf.broadcast_to(tf.expand_dims(x, axis=1), shape=(N, N, embed_size))
    eTx = tf.broadcast_to(tf.expand_dims(x, axis=0), shape=(N, N, embed_size))

    dists = tf.reduce_sum(tf.square(xTe - eTx), axis=2)
    z = zeros([N, N], dtype=DTYPE)

    W_dists = tf.where(W, dists, z)
    Wp_dists = tf.where(Wp, dists, z)

    return W_dists, Wp_dists


def filt_k_max(dists, k):
    N = tf.shape(dists)[0]
    k = tf.minimum(tf.constant(k, dtype=tf.int32), N)
    vals, inds = tf.nn.top_k(dists, k=k)
    inds = tf.where(
        tf.greater(vals, tf.zeros_like(vals, dtype=DTYPE)), inds, -tf.ones_like(inds)
    )
    inds = tf.sparse.to_indicator(
        sp_input=tf.sparse.from_dense(inds + 1), vocab_size=N + 1
    )[:, 1:]
    filt_dists = tf.where(inds, dists, tf.zeros_like(dists, dtype=DTYPE))
    return tf.maximum(filt_dists, tf.transpose(filt_dists))


def filt_k_min(dists, k):
    N = tf.shape(dists)[0]
    k = tf.minimum(tf.constant(int(k), dtype=tf.int32), N)

    discarded = tf.multiply(
        tf.constant(DTYPE.max, dtype=DTYPE), tf.ones_like(dists, dtype=DTYPE)
    )
    neg_dists = -tf.where(
        tf.equal(dists, tf.zeros_like(dists, dtype=DTYPE)), discarded, dists
    )
    vals, inds = tf.nn.top_k(neg_dists, k=k)
    inds = tf.where(
        tf.less(vals, tf.zeros_like(vals, dtype=DTYPE)), inds, -tf.ones_like(inds)
    )
    inds = tf.sparse.to_indicator(
        sp_input=tf.sparse.from_dense(inds + 1), vocab_size=N + 1
    )[:, 1:]
    filt_dists = tf.where(inds, dists, tf.zeros_like(dists, dtype=DTYPE))
    return tf.maximum(filt_dists, tf.transpose(filt_dists))


def filt_k_min_any(dists, k):
    N = tf.shape(dists)[0]
    k = tf.minimum(tf.constant(int(k), dtype=tf.int32), N * N)
    orig_shape = tf.shape(dists)

    # select only the upper diagonal
    upper = tf.matrix_band_part(dists, 0, -1) - tf.diag(tf.diag_part(dists))

    # reshape to vector
    flattened = tf.reshape(upper, [-1, tf.math.reduce_prod(orig_shape)])

    # find k smalles
    discarded = tf.multiply(
        tf.constant(DTYPE.max, dtype=DTYPE), tf.ones_like(flattened, dtype=DTYPE)
    )
    neg_flat = -tf.where(
        tf.equal(flattened, tf.zeros_like(flattened, dtype=DTYPE)), discarded, flattened
    )

    # filter k smallest
    _, inds_flat = tf.nn.top_k(neg_flat, k=k)

    # transform into dense indicator
    inds_dense = tf.sparse.to_indicator(
        sp_input=tf.sparse.from_dense(inds_flat), vocab_size=tf.shape(flattened)[-1]
    )

    inds_upper = tf.cast(tf.reshape(inds_dense, orig_shape), dtype=DTYPE)
    inds = tf.cast(tf.add(inds_upper, tf.transpose(inds_upper)), dtype=tf.bool)

    return tf.where(inds, dists, tf.zeros_like(dists, dtype=DTYPE))


def filt_epsilon(x, eps):
    eps = tf.multiply(tf.constant(eps, dtype=DTYPE), tf.ones_like(x, dtype=DTYPE))
    return tf.where(tf.less(x, eps), x, tf.zeros_like(x, dtype=DTYPE))


# Weight types
def dist2indicator(x):
    ii = tf.ones_like(x, dtype=DTYPE)
    oo = tf.zeros_like(x, dtype=DTYPE)
    return tf.where(tf.greater(x, oo), ii, oo)


def dist2gaussian(x):
    gaussian = tf.math.exp(-x)
    oo = tf.zeros_like(x, dtype=DTYPE)
    # return tf.where(tf.equal(x, O), O, gaussian)
    return tf.where(tf.greater(gaussian, oo), gaussian, oo)


# FilterType
def filter_all(W_Wp, xs=None, xt=None):
    return W_Wp


def make_filter(
    filter_type: FilterType,
    penalty_filter_type: FilterType,
    filter_param: int,
    penalty_filter_param: int,
):
    filt_dict = {
        FilterType.ALL: lambda x, p: x,
        FilterType.KNN: filt_k_min,
        FilterType.KFN: filt_k_max,
        FilterType.KSD: filt_k_min_any,
        FilterType.EPSILON: filt_epsilon,
    }
    filt_fn = filt_dict[filter_type]
    p_filt_fn = filt_dict[penalty_filter_type]

    def fn(W_Wp, xs, xt):
        dists, p_dists = get_filt_dists(W_Wp, xs, xt)
        dists = filt_fn(dists, filter_param)
        p_dists = p_filt_fn(p_dists, penalty_filter_param)
        return dists, p_dists

    return fn


# ConnectionTypes
def connect_all(ys, yt, batch_size):
    N = 2 * batch_size
    y = tf.concat([ys, yt], axis=0)
    yTe = tf.broadcast_to(tf.expand_dims(y, axis=1), shape=(N, N))
    eTy = tf.broadcast_to(tf.expand_dims(y, axis=0), shape=(N, N))

    W = tf.equal(yTe, eTy)
    Wp = tf.not_equal(yTe, eTy)

    return W, Wp


def connect_source_target(ys, yt, batch_size, intrinsic=True, penalty=True):
    W, Wp = connect_all(ys, yt, batch_size)

    N = 2 * batch_size
    tile_size = tf.repeat(batch_size, 2)
    oo = zeros(tile_size, dtype=tf.bool)
    ii = ones(tile_size, dtype=tf.bool)
    ind = tf.concat([tf.concat([oo, ii], axis=0), tf.concat([ii, oo], axis=0)], axis=1)
    oo = zeros(tf.repeat(N, 2), dtype=tf.bool)

    if intrinsic:
        W = tf.where(ind, W, oo)
    if penalty:
        Wp = tf.where(ind, Wp, oo)

    return W, Wp


def connect_source_target_pair(ys, yt, batch_size):
    eq = tf.linalg.diag(tf.equal(ys, yt))
    neq = tf.linalg.diag(tf.not_equal(ys, yt))
    oo = zeros(tf.repeat(batch_size, 2), dtype=tf.bool)
    W = tf.concat([tf.concat([oo, eq], axis=0), tf.concat([eq, oo], axis=0)], axis=1)
    Wp = tf.concat([tf.concat([oo, neq], axis=0), tf.concat([neq, oo], axis=0)], axis=1)
    return W, Wp


# Loss maker
def dage_loss(
    connection_type: ConnectionType,
    weight_type: WeightType,
    filter_type: FilterType,
    penalty_filter_type: FilterType,
    filter_param=1,
    penalty_filter_param=1,
):
    connection_type = str2enum(connection_type, ConnectionType)
    weight_type = str2enum(weight_type, WeightType)
    filter_type = str2enum(filter_type, FilterType)
    penalty_filter_type = str2enum(penalty_filter_type, FilterType)

    connect = {
        ConnectionType.ALL: connect_all,
        ConnectionType.SOURCE_TARGET: connect_source_target,
        ConnectionType.SOURCE_TARGET_PAIR: connect_source_target_pair,
        ConnectionType.ST_INT_ALL_PEN: partial(
            connect_source_target, intrinsic=True, penalty=False
        ),
    }[connection_type]

    transform = {
        WeightType.INDICATOR: lambda W_Wp: (
            dist2indicator(W_Wp[0]),
            dist2indicator(W_Wp[1]),
        ),
        WeightType.GAUSSIAN: lambda W_Wp: (
            dist2gaussian(W_Wp[0]),
            dist2gaussian(W_Wp[1]),
        ),
    }[weight_type]

    filt = make_filter(
        filter_type, penalty_filter_type, filter_param, penalty_filter_param
    )

    if (
        filter_type == FilterType.ALL
        and penalty_filter_type == FilterType.ALL
        and weight_type == WeightType.INDICATOR
    ):
        # performance optimisation
        def make_weights(xs, xt, ys, yt, batch_size):
            W, Wp = connect(ys, yt, batch_size)
            return tf.cast(W, dtype=DTYPE), tf.cast(Wp, dtype=DTYPE)

    else:

        def make_weights(xs, xt, ys, yt, batch_size):
            W, Wp = transform(filt(W_Wp=connect(ys, yt, batch_size), xs=xs, xt=xt))
            return tf.cast(W, dtype=DTYPE), tf.cast(Wp, dtype=DTYPE)

    def loss_fn(y_true, y_pred):
        ys = tf.argmax(tf.cast(y_true[:, 0], dtype=tf.int32), axis=1)
        yt = tf.argmax(tf.cast(y_true[:, 1], dtype=tf.int32), axis=1)
        xs = y_pred[:, 0]
        xt = y_pred[:, 1]
        θϕ = tf.transpose(tf.concat([xs, xt], axis=0))

        bs = tf.shape(ys)[0]

        # construct Weight matrix
        W, Wp = make_weights(xs, xt, ys, yt, bs)

        # construct Degree matrix
        D = tf.linalg.diag(tf.reduce_sum(W, axis=1))
        Dp = tf.linalg.diag(tf.reduce_sum(Wp, axis=1))

        # construct Graph Laplacian
        L = tf.subtract(D, W)
        Lp = tf.subtract(Dp, Wp)

        # construct loss
        θϕLϕθ = tf.matmul(θϕ, tf.matmul(L, θϕ, transpose_b=True))
        θϕLpϕθ = tf.matmul(θϕ, tf.matmul(Lp, θϕ, transpose_b=True))

        loss = tf.linalg.trace(θϕLϕθ) / (tf.linalg.trace(θϕLpϕθ) + 1e-11)

        return loss

    return loss_fn
