import tensorflow as tf
keras = tf.compat.v2.keras

def model_dense(input_shape, dense_size, embed_size):
    return keras.Sequential([
        keras.layers.Input(shape=input_shape),
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


def model_preds(input_shape, output_shape):
    return keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(output_shape, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='logits'),
        keras.layers.Activation('softmax', name='preds'),
    ], name='preds')


def model_logits(input_shape, dense_size):
    return keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(dense_size, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='logits'),
    ], name='logits')


def get_output_shape(model):
    output_shape = model.layers[-1].output_shape
    if type(output_shape) == list:
        output_shape = output_shape[0]
    if type(output_shape) == tuple:
        output_shape = output_shape[1:]
    return output_shape


def freeze(model, num_leave_unfrozen=0):
    for layer in model.layers[:-num_leave_unfrozen]:
            layer.trainable = False
