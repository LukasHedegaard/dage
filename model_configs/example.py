import neural_net as nn
import tensorflow as tf

layers = tf.keras.layers


def build(doc_length, n_classes, embedding_weight_loader):
    input_layer, embedding_layer, embedding_size = nn.build_init_layers(
        doc_length=doc_length,
        embedding_weight_loader=embedding_weight_loader,
        reshape=True
    )

    x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', name='conv-{0}'.format(1))(embedding_layer)
    x = layers.MaxPool2D(pool_size=(2, 2), name='conv-{0}-maxpool'.format(1))(x)
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', name='conv-{0}'.format(2))(x)
    x = layers.MaxPool2D(pool_size=(2, 2), name='conv-{0}-maxpool'.format(2))(x)
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', name='conv-{0}'.format(3))(x)
    x = layers.MaxPool2D(pool_size=(2, 2), name='conv-{0}-maxpool'.format(3))(x)
    x = layers.Flatten()(x)

    output = tf.keras.layers.Dense(n_classes, activation='softmax', name='prediction')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output)

    return model
