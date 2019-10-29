import tensorflow as tf
M = tf.compat.v2.math
keras = tf.compat.v2.keras
K = keras.backend
from utils.dataset_gen import DTYPE
from math import ceil

class Distance(keras.layers.Layer):
    '''
    Distance between input elements.
    Assumes the input two be array-like with two elements
    '''
    def call(self, inputs):
        diff = M.subtract(inputs[0], inputs[1])
        dist = M.sqrt(M.reduce_sum(M.square(diff),axis=1)) # frobenius norm
        return dist


def contrastive_loss(indicator, distance, margin:float=100):
    indicator = tf.cast(indicator, DTYPE)
    m = tf.constant(margin, dtype=DTYPE)

    # loss for similarity
    L_S = M.multiply(
        tf.constant(0.5, dtype=DTYPE), 
        M.square(distance)
    )
    # loss for dissimilarity
    L_D = M.multiply(
        tf.constant(0.5, dtype=DTYPE), 
        M.square(M.maximum(
            tf.constant(0, dtype=DTYPE), 
            M.subtract(m, distance)
        ))
    )
    # resulting contrastive loss
    L_cont = M.add(
        M.multiply(M.subtract(m, distance), L_S),
        M.multiply(indicator, L_D)
    )
    return M.reduce_mean(L_cont) # average over batch


class CCSAModel:
    def __init__(self, 
        model_base, 
        output_shape, 
        freeze_base=True,
        alpha=0.01,
    ):
        input_shape = model_base.layers[0].input_shape[0][1:]
        in1 = keras.layers.Input(shape=input_shape, name='input_source')
        in2 = keras.layers.Input(shape=input_shape, name='input_target')

        self.model_base = model_base
        if freeze_base:
            for layer in self.model_base.layers:
                layer.trainable = False

        self.model_mid = keras.Sequential([
            keras.layers.Input(shape=self.model_base.layers[-1].output_shape[1:]),
            keras.layers.Dropout(0.25),
            keras.layers.Flatten(),
            keras.layers.Dense(1024, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', name='dense'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', name='embed'),
        ], name='dense_layers')

        self.model_top = keras.Sequential([
            keras.layers.Input(shape=self.model_mid.layers[-1].output_shape[1:]),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(output_shape, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='logits'),
            keras.layers.Activation('softmax', name='preds'),
        ], name='preds')

        # weight sharing is used: the same instance of model_base, and model_mid is used for both streams
        mid1 = self.model_mid(self.model_base(in1))
        mid2 = self.model_mid(self.model_base(in2))

        # the original authors had only a single prediction output, and had to feed every batch twice, flipping source and target on the second run
        # we instead create two prediction layers (shared weights) as a performance optimisation (base and mid only run once)
        out1 = self.model_top(mid1)
        out2 = self.model_top(mid2)

        aux_out = Distance(name='aux_out')([mid1, mid2])

        self.model = keras.models.Model(
            inputs=[in1, in2],
            outputs=[out1, out2, aux_out]
        )

        self.loss = {
            'preds'  : keras.losses.categorical_crossentropy,
            'preds_1': keras.losses.categorical_crossentropy,
            'aux_out': contrastive_loss
        }

        self.loss_weights = {
            'preds'  : 0.5 * (1-alpha),
            'preds_1': 0.5 * (1-alpha),
            'aux_out': alpha
        }

    def train(self,
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


