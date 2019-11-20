import tensorflow as tf
from utils.dataset_gen import DTYPE

# def padded(t, shape, pad_val=0):
#     def pad_up_to(t, max_in_dims, constant_values=0):
#         s = tf.shape(t)
#         paddings = [[0, m-s[i]] for (i,m) in enumerate(max_in_dims)]
#         return tf.pad(t, paddings, 'CONSTANT', constant_values=0)

#     return tf.cond(tf.reduce_any(tf.less(tf.shape(t), shape)), true_fn=lambda: pad_up_to(t, shape, pad_val), false_fn=lambda: t)

# def pad_to_batch(t, batch_size, pad_val=0):
#     shape = (batch_size, *[s.value for s in t.shape[1:]])
#     return padded(t,shape,pad_val)


def dnse_loss(
    margin=1
):
    def loss(y_true, y_pred):
        ''' Tensorflow implementation of d-SNE loss. 
            Original Mxnet implementation found at https://github.com/aws-samples/d-SNE.
            @param y_true: tuple or array of two elements, containing source and target features
            @param y_pred: tuple or array of two elements, containing source and taget labels
        '''
        xs = y_pred[:,0]
        xt = y_pred[:,1]
        ys = tf.argmax(tf.cast(y_true[:,0], dtype=tf.int32), axis=1)
        yt = tf.argmax(tf.cast(y_true[:,1], dtype=tf.int32), axis=1)

        batch_size = tf.shape(ys)[0]
        embed_size = tf.shape(xs)[1]

        # The original implementation provided an optional feature-normalisation (L2) here. We'll skip it

        xs_rpt = tf.broadcast_to(tf.expand_dims(xs, axis=0), shape=(batch_size, batch_size, embed_size))
        xt_rpt = tf.broadcast_to(tf.expand_dims(xt, axis=1), shape=(batch_size, batch_size, embed_size))

        dists = tf.reduce_sum(tf.square(xt_rpt - xs_rpt), axis=2)

        yt_rpt = tf.broadcast_to(tf.expand_dims(yt, axis=1), shape=(batch_size, batch_size))
        ys_rpt = tf.broadcast_to(tf.expand_dims(ys, axis=0), shape=(batch_size, batch_size))

        y_same = tf.equal(yt_rpt, ys_rpt)
        y_diff = tf.not_equal(yt_rpt, ys_rpt)

        intra_cls_dists = tf.multiply(dists, tf.cast(y_same, dtype=DTYPE))
        inter_cls_dists = tf.multiply(dists, tf.cast(y_diff, dtype=DTYPE))

        max_dists = tf.reduce_max(dists, axis=1, keepdims=True)
        max_dists = tf.broadcast_to(max_dists, shape=(batch_size, batch_size))
        revised_inter_cls_dists = tf.where(y_same, max_dists, inter_cls_dists)

        max_intra_cls_dist = tf.reduce_max(intra_cls_dists, axis=1)
        min_inter_cls_dist = tf.reduce_min(revised_inter_cls_dists, axis=1)

        loss = tf.nn.relu(max_intra_cls_dist - min_inter_cls_dist + margin)

        return loss

    return loss