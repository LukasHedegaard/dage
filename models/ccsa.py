import tensorflow as tf
M = tf.compat.v2.math
keras = tf.compat.v2.keras
K = keras.backend
from utils.dataset_gen import DTYPE
from math import ceil
from functools import reduce

class Diff(keras.layers.Layer):
    '''
    Distance between input elements.
    Assumes the input two be array-like with two elements
    '''
    def call(self, inputs):
        diff = M.subtract(inputs[0], inputs[1])
        return diff


def contrastive_loss(indicator, diff, margin:float=1):
    s = tf.cast(indicator, DTYPE) # similarity indicator
    m = tf.constant(margin, dtype=DTYPE)
    d = M.sqrt(M.reduce_sum(M.square(diff),axis=1))

    # loss for similarity
    L_S = M.multiply(
        tf.constant(0.5, dtype=DTYPE), 
        M.square(d)
    )
    # loss for dissimilarity
    L_D = M.multiply(
        tf.constant(0.5, dtype=DTYPE), 
        M.square(M.maximum(
            tf.constant(0, dtype=DTYPE), 
            M.subtract(m, d)
        ))
    )
    # resulting contrastive loss
    L_cont = M.add(
        M.multiply(s, L_S),
        M.multiply(M.subtract(tf.constant(1, dtype=DTYPE), s), L_D)
    )
    return M.reduce_mean(L_cont) # average over batch


def euclidean_distance(vects):
    eps = 1e-08
    x, y = vects  # Shapes of the vectors are [batch_size, feature_extractor_size]
    squared_diff = K.square(x - y)
    sum_of_squares = K.sum(squared_diff, axis=1, keepdims=True)  # shape=[batch_size, 1]
    max_val = K.maximum(sum_of_squares, eps)  # shape=[batch_size, 1]
    dist = K.sqrt(max_val)

    # return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), eps))
    return dist

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss_orig(y_true, dist):
    margin = 1
    return K.mean(y_true * K.square(dist) + (1 - y_true) * K.square(K.maximum(margin - dist, 0)))


def model(
    model_base, 
    input_shape,
    output_shape, 
    freeze_base=True,
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

    model_mid = keras.Sequential([
        keras.layers.Input(shape=model_base.layers[-1].output_shape[1:]),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        # keras.layers.Dense(1024, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', name='dense'),
        # keras.layers.Dense(1024, activation='relu', activity_regularizer=keras.regularizers.l2(0.01), kernel_initializer='glorot_uniform', bias_initializer='zeros', name='dense'),
        keras.layers.Dense(1024, activation=None, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='dense'),
        keras.layers.BatchNormalization(momentum=0.9),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.5),
        # keras.layers.Dense(128, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', name='embed'),
        # keras.layers.Dense(128, activation='relu', activity_regularizer=keras.regularizers.l2(0.01), kernel_initializer='glorot_uniform', bias_initializer='zeros', name='embed'),
        keras.layers.Dense(128, activation=None, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='embed'),
        keras.layers.BatchNormalization(momentum=0.9),
        keras.layers.Activation('relu'),
    ], name='dense_layers')

    # for layer in model_mid.layers[:-3]:
    #     layer.trainable = False

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

    # aux_out = Diff(name='aux_out')([mid1, mid2])
    aux_out = keras.layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape, name='aux_out')([mid1, mid2])

    model = keras.models.Model(
        inputs=[in1, in2],
        outputs=[out1, out2, aux_out]
    )
    return model

def loss():
    return {
        'preds'  : keras.losses.categorical_crossentropy,
        'preds_1': keras.losses.categorical_crossentropy,
        'aux_out': contrastive_loss_orig
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
    ''' ccsa training procedure
    '''
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

## The below code is leftover from porting the original ccsa implementation

def orig_base_model(): 
    img_rows, img_cols = 224, 224
    nb_filters = 32
    pool_size = (2, 2)
    kernel_size = (3, 3)
    input_shape = (img_rows, img_cols, 3)

    model = keras.Sequential()
    model.add(keras.layers.Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),
                            padding='valid',
                            input_shape=input_shape))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Convolution2D(nb_filters, (kernel_size[0], kernel_size[1])))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))

    # model.add(keras.layers.Dropout(0.25))
    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(120))
    # model.add(keras.layers.Activation('relu'))
    # model.add(keras.layers.Dense(84))
    # model.add(keras.layers.Activation('relu'))
    return model

def orig_model(
    model_base, 
    output_shape, 
    freeze_base=True,
    alpha=0.01,
):
    # size of digits 16*16
    img_rows, img_cols = 224, 224
    input_shape = (img_rows, img_cols, 3)
    in1 = keras.layers.Input(shape=input_shape, name='input_source')
    in2 = keras.layers.Input(shape=input_shape, name='input_target')

    if freeze_base:
        for layer in model_base.layers:
            layer.trainable = False

    model_mid = keras.Sequential([
        keras.layers.Input(shape=model_base.layers[-1].output_shape[1:]),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(120, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', name='dense'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(84, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', name='embed'),
    ], name='dense_layers')

    mid1 = model_mid(model_base(in1))
    mid2 = model_mid(model_base(in2))

    model_top = keras.Sequential([
        keras.layers.Input(shape=model_mid.layers[-1].output_shape[1:]),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(output_shape, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='logits'),
        keras.layers.Activation('softmax', name='preds'),
    ], name='preds')

    out1 = model_top(mid1)
    out2 = model_top(mid2)

    distance = keras.layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape, name='aux_out')([mid1, mid2])
    model = keras.models.Model(inputs=[in1, in2], outputs=[out1, out2, distance])
    return model

def orig_loss():
    return {
        'preds': 'categorical_crossentropy', 
        'preds_1': 'categorical_crossentropy', 
        'aux_out': contrastive_loss
    }

def orig_loss_weights(alpha=0.25):
    return {
        'preds': 1*(1-alpha), 
        'preds_1': 0*(1-alpha), 
        'aux_out': alpha
    }

def orig_train(
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
    
    for e in range(epochs):
        if verbose:
            print('Epoch {}/{}.'.format(e, epochs))

        for step in range(steps_per_epoch):
            ins, outs = next(train_iter)

            X1 = ins['input_source']
            X2 = ins['input_target']
            y1 = outs['preds']
            y2 = outs['preds_1']
            yc = outs['aux_out']

            source_loss = model.train_on_batch([X1, X2],
                                               [y1, y2, yc])

            if step % 10 == 0 and verbose:
                print(' Step {}/{}'.format(step, steps_per_epoch))
                print('  Source Pass:  {}'.format(
                    '  '.join(['{} {:0.4f}'.format(model.metrics_names[i], source_loss[i]) for i in range(len(source_loss))])
                ))

        val_iter = iter(val_datasource)
        val_loss = []
        for step in range(validation_steps):
            ins, outs = next(val_iter)
            X1 = ins['input_source']
            X2 = ins['input_target']
            y1 = outs['preds']
            y2 = outs['preds_1']
            yc = outs['aux_out']
            val_loss.append(model.test_on_batch([X1, X2],
                                                [y1, y2, yc]))

        val_loss_avg = reduce(
            lambda n, o: [n[i]+o[i] for i in range(len(val_loss))],
            val_loss, 
            [0 for _ in val_loss[0]]
        )

        if verbose:
            print('  Validation:  {}'.format(
                '  '.join(['{} {:0.4f}'.format(model.metrics_names[i], val_loss_avg[i]) for i in range(len(val_loss_avg))])
            ))


def orig_train_flipping(
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
    
    for e in range(epochs):
        if verbose:
            print('Epoch {}/{}.'.format(e, epochs))

        for step in range(steps_per_epoch):
            ins, outs = next(train_iter)

            X1 = ins['input_source']
            X2 = ins['input_target']
            y1 = outs['preds']
            y2 = outs['preds_1']
            yc = outs['aux_out']

            source_loss = model.train_on_batch([X1, X2],
                                               [y1, y2, yc])

            target_loss = model.train_on_batch([X2, X1],
                                               [y2, y1, yc])

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
            X1 = ins['input_source']
            X2 = ins['input_target']
            y1 = outs['preds']
            y2 = outs['preds_1']
            yc = outs['aux_out']
            val_loss.append(model.test_on_batch([X1, X2],
                                                [y1, y2, yc]))

        val_loss_avg = reduce(
            lambda n, o: [n[i]+o[i] for i in range(len(val_loss))],
            val_loss, 
            [0 for _ in val_loss[0]]
        )

        if verbose:
            print('  Validation:  {}'.format(
                '  '.join(['{} {:0.4f}'.format(model.metrics_names[i], val_loss_avg[i]) for i in range(len(val_loss_avg))])
            ))


