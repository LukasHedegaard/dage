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

def euclidean_distance(x1, x2):
    return K.sqrt(K.maximum(K.sum(K.square(x1-x2), axis=1, keepdims=False), 1e-08))

def labels_equal(y1, y2):
    return tf.cast(tf.reduce_all(tf.equal(y1,y2), axis=1, keepdims=False), dtype=tf.float32)

def contrastive_loss(#self, 
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
    # return K.mean(losses)
    return losses


def model(
    model_base, 
    input_shape,
    output_shape,
    optimizer,
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
        loss=loss(), 
        loss_weights=loss_weights(alpha, even_loss_weights), 
        optimizer=optimizer, 
        metrics={'preds':'accuracy', 'preds_1':'accuracy'},
    )

    return model

def loss():
    return {
        'preds'  : keras.losses.categorical_crossentropy,
        'preds_1': keras.losses.categorical_crossentropy,
        'aux_out': contrastive_loss
    }

def loss_weights(alpha=0.25, even=True):
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
    validation_steps = ceil(val_datasource_size/batch_size) if val_datasource_size else None
    steps_per_epoch = ceil(datasource_size/batch_size)

    train_iter = iter(datasource)
    
    for e in range(1,epochs+1):
        if verbose:
            print('Epoch {}/{}.'.format(e, epochs))

        for step in range(steps_per_epoch):
            ins, outs = next(train_iter)

            source_loss = model.train_on_batch(ins, outs)

            target_loss = model.train_on_batch(
                {'input_source': ins['input_target'], 'input_target': ins['input_source']},
                {'preds': outs['preds_1'], 'preds_1':outs['preds'], 'aux_out': outs['aux_out']}
            )

            if step % 10 == 0 and verbose:
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
            ins, outs = next(val_iter)
            val_loss.append(model.test_on_batch(ins, outs))

        val_loss_avg = reduce(
            lambda n, o: [n[i]+o[i] for i in range(len(val_loss))],
            val_loss, 
            [0 for _ in val_loss[0]]
        )

        if verbose:
            print('  Validation:  {}'.format(
                '  '.join(['{} {:0.4f}'.format(model.metrics_names[i], val_loss_avg[i]) for i in range(len(val_loss_avg))])
            ))
