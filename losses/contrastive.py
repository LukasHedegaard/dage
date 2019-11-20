import tensorflow as tf
K = tf.compat.v2.keras.backend
from utils.dataset_gen import DTYPE

def euclidean_distance(x1, x2):
    return K.sqrt(K.maximum(K.sum(K.square(x1-x2), axis=1, keepdims=False), 1e-08))

def labels_equal(y1, y2):
    return tf.cast(tf.reduce_all(tf.equal(y1,y2), axis=1, keepdims=False), dtype=DTYPE)

def contrastive_loss(
    y_true, 
    y_pred
):
    ''' Implementation of contrastive loss. 
        Original implementation found at https://github.com/samotiian/CCSA
        @param y_true: distance between source and target features
        @param y_pred: tuple or array of two elements, containing source and taget labels
    '''
    margin = 1
    xs, xt = y_pred[:,0], y_pred[:,1]
    ys, yt = y_true[:,0], y_true[:,1]
    
    dist = euclidean_distance(xs, xt)
    label = labels_equal(ys, yt)

    losses = label * K.square(dist) + (1 - label) * K.square(K.maximum(margin - dist, 0))
    return losses