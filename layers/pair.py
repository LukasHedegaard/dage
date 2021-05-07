import tensorflow as tf

keras = tf.compat.v2.keras


def Pair(name, embed_size, num_in_pair=2):
    """Collects the input layers into a pair.
    Assummes that input dimensions of the inputs equal.
    """
    layers = [
        keras.layers.Concatenate(axis=1),
        keras.layers.Reshape(
            (num_in_pair, embed_size),
            input_shape=(num_in_pair * embed_size,),
            name=name,
        ),
    ]

    def call(x):
        for layer in layers:
            x = layer(x)
        return x

    return call
