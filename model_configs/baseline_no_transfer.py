import tensorflow as tf

layers = tf.keras.layers

def build(input_shape, n_classes):

    i = tf.keras.layers.Input(shape=input_shape, name='input')
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(i)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    o = layers.Dense(n_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=i, outputs=o)

    return model
