import tensorflow as tf
M = tf.compat.v2.math
keras = tf.compat.v2.keras
K = keras.backend
from utils.dataset_gen import DTYPE

class Distance(keras.layers.Layer):
    '''
    Frobenius Norm of distance between inputs.
    Assumes the input two be array-like with two elements
    Implementation only supports 1D data
    '''
    def __init__(self, name=None):
        super(Distance, self).__init__(name=name)

    def call(self, inputs):
        a = inputs[0]
        b = inputs[1]
        return M.square(M.sqrt(M.reduce_sum(M.square(a-b),axis=1)))


def csa_loss(frob_norm, indicator):
    m = tf.constant(1, dtype=DTYPE)                                             # margin
    d = M.multiply(0.5, M.square(frob_norm))                                    # distance metric
    k = M.multiply(0.5, M.square(M.maximum(tf.constant(0, dtype=DTYPE), M.subtract(m, frob_norm))))   # similarity metric
    L_SA = M.multiply(indicator, d)                                             # semantic allignment loss
    L_S  = M.multiply(M.subtract(tf.constant(1, dtype=DTYPE), indicator), k)    # separation loss
    L_CSA = L_SA + L_S                                                          # contrastive semantic alligment loss
    return L_CSA


class CCSAModel:
    def __init__(self, model_base, output_shape, alpha=0.25):
        input_shape = model_base.layers[0].input_shape[0][1:]
        in1 = keras.layers.Input(shape=input_shape, name='input_source')
        in2 = keras.layers.Input(shape=input_shape, name='input_target')

        self.model_base = model_base

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
        ], name='pred')

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
            'pred'    : keras.losses.categorical_crossentropy,
            'pred_1'  : keras.losses.categorical_crossentropy,
            'aux_out' : csa_loss
        }

        self.loss_weights = {
            'pred'    : 0.5 * (1-alpha),
            'pred_1'  : 0.5 * (1-alpha),
            'aux_out' : alpha
        }
        
    def _set_trainable_top(self):
        for layer in self.model_base.layers:
            layer.trainable = False
        for layer in self.model_mid.layers:
            layer.trainable = True 
        for layer in self.model_top.layers:
            layer.trainable = True 

    def _set_trainable_mid(self, num_base_layers_to_train=4):
        for layer in self.model_base.layers[:-num_base_layers_to_train]:
            layer.trainable = False
        for layer in self.model_base.layers[:-num_base_layers_to_train]:
            layer.trainable = True
        for layer in self.model_mid.layers:
            layer.trainable = True 
        for layer in self.model_top.layers:
            layer.trainable = True 

    def _set_trainable_all(self):
        for layer in self.model_base.layers:
            layer.trainable = True
        for layer in self.model_mid.layers:
            layer.trainable = True 
        for layer in self.model_top.layers:
            layer.trainable = True 

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
        ''' fine-tuning procedure
        '''
        validation_steps = val_datasource_size//batch_size if val_datasource_size else None

        if verbose:
            print('Training top only')
        self._set_trainable_top()
        model.compile(loss=model.loss, loss_weights=model.loss_weights, optimizer=model.optimizer, metrics=model.metrics)
        model.fit( 
            x=datasource, 
            validation_data=val_datasource,
            epochs=epochs//2, 
            steps_per_epoch=datasource_size//batch_size, 
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=verbose,
        )

        if verbose:
            print('Training top and base top {} layers'.format(4))
        self._set_trainable_mid(num_base_layers_to_train=4)
        model.compile(loss=model.loss, loss_weights=model.loss_weights, optimizer=model.optimizer, metrics=model.metrics)
        model.fit( 
            x=datasource, 
            validation_data=val_datasource,
            epochs=epochs//2, 
            steps_per_epoch=datasource_size//batch_size, 
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=verbose,
        )

        if verbose:
            print('Training whole network')
        self._set_trainable_all()
        model.compile(loss=model.loss, loss_weights=model.loss_weights, optimizer=model.optimizer, metrics=model.metrics)
        model.fit( 
            x=datasource, 
            validation_data=val_datasource,
            epochs=epochs//2, 
            steps_per_epoch=datasource_size//batch_size, 
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=verbose,
        )
    
