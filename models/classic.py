import tensorflow as tf
keras = tf.compat.v2.keras

def model(model_base, output_shape):
    return keras.Sequential([
        model_base,
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', name='dense'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(512, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', name='embed'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(output_shape, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='logits'),
        keras.layers.Activation('softmax', name='preds'),
    ])