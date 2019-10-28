import tensorflow as tf
keras = tf.compat.v2.keras
from math import ceil

def model(
    model_base, 
    output_shape,
    freeze_base=True,
):
    if freeze_base:
        for layer in self.model_base.layers:
            layer.trainable = False

    model = keras.Sequential([
        model_base,
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', name='dense'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', name='embed'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(output_shape, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='logits'),
        keras.layers.Activation('softmax', name='preds'),
    ])

    return model


def trainable_top(model):
    model.layers[0].trainable = False
    for layer in model.layers[1:]:
        layer.trainable = True

    return model


def trainable_base_top(model, num_base_layers_to_train=4):
    for layer in model.layers[0].layers[:-num_base_layers_to_train]:
        layer.trainable = False

    for layer in model.layers[0].layers[-num_base_layers_to_train:]:
        layer.trainable = True

    for layer in model.layers[1:]:
        layer.trainable = True 

    return model


def trainable(model):
    for layer in model.layers[0].layers:
        layer.trainable = True

    for layer in model.layers[1:]:
        layer.trainable = True 

    return model


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
    ''' fine-tuning procedure
    '''

    validation_steps = ceil(val_datasource_size/batch_size) if val_datasource_size else None
    steps_per_epoch = ceil(datasource_size/batch_size)

    model.fit( 
        x=datasource, 
        validation_data=val_datasource,
        epochs=epochs, 
        steps_per_epoch=steps_per_epoch, 
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=verbose,
    )

    # if verbose:
    #     print('Training top only')
    # model = trainable_top(model)
    # model.compile(loss=model.loss, loss_weights=model.loss_weights, optimizer=model.optimizer, metrics=model.metrics)
    # model.fit( 
    #     x=datasource, 
    #     validation_data=val_datasource,
    #     epochs=epochs//3, 
    #     steps_per_epoch=steps_per_epoch, 
    #     validation_steps=validation_steps,
    #     callbacks=callbacks,
    #     verbose=verbose,
    # )
    # if verbose:
    #     print('Training top and base top {} layers'.format(4))
    # model = trainable_base_top(model, 4)
    # model.compile(loss=model.loss, loss_weights=model.loss_weights, optimizer=model.optimizer, metrics=model.metrics)
    # model.fit( 
    #     x=datasource, 
    #     validation_data=val_datasource,
    #     epochs=epochs//3, 
    #     steps_per_epoch=steps_per_epoch, 
    #     validation_steps=validation_steps,
    #     callbacks=callbacks,
    #     verbose=verbose,
    # )
    # if verbose:
    #     print('Training whole network')
    # model = trainable(model)
    # model.compile(loss=model.loss, loss_weights=model.loss_weights, optimizer=model.optimizer, metrics=model.metrics)
    # model.fit( 
    #     x=datasource, 
    #     validation_data=val_datasource,
    #     epochs=epochs//3, 
    #     steps_per_epoch=steps_per_epoch, 
    #     validation_steps=validation_steps,
    #     callbacks=callbacks,
    #     verbose=verbose,
    # )