from functools import reduce
from math import ceil

import tensorflow as tf

from layers import AngularLinear, L2NormInstance
from utils.cyclical_learning_rate import CyclicLR

keras = tf.compat.v2.keras
K = keras.backend
DTYPE = tf.float32


def get_output_shape(model):
    output_shape = model.layers[-1].output_shape
    if type(output_shape) == list:
        output_shape = output_shape[0]
    if type(output_shape) == tuple:
        output_shape = output_shape[1:]
    return output_shape


def freeze(model, num_leave_unfrozen=0):
    if num_leave_unfrozen == -1:
        return model

    if num_leave_unfrozen == 0:
        for layer in model.layers:
            layer.trainable = False
    else:
        for layer in model.layers[:-num_leave_unfrozen]:
            layer.trainable = False
        if num_leave_unfrozen != 0:
            for layer in model.layers[-num_leave_unfrozen:]:
                layer.trainable = True


def conv2_block(input_shape, l2=0.0001, dropout=0.5, batch_norm=False):  # input only
    i = keras.layers.Input(shape=input_shape)
    o = i

    if batch_norm:
        o = keras.layers.BatchNormalization(momentum=0.9)(o)

    o = keras.layers.Conv2D(6, (5, 5), kernel_regularizer=keras.regularizers.l2(l=l2))(
        o
    )
    o = keras.layers.Activation("relu")(o)
    if dropout:
        o = keras.layers.Dropout(dropout)(o)
    o = keras.layers.Conv2D(16, (5, 5), kernel_regularizer=keras.regularizers.l2(l=l2))(
        o
    )
    o = keras.layers.Activation("relu")(o)
    o = keras.layers.MaxPool2D()(o)

    model = keras.models.Model(inputs=[i], outputs=[o], name="embeds")
    return model


def lenetplus_conv_block(
    input_shape,
    l2=0.0001,
    dropout=0.5,
    num_filters=[32, 64, 128],
    batch_norm=False,
):
    i = keras.layers.Input(shape=input_shape)
    o = i

    for filters in num_filters:
        if batch_norm:
            o = keras.layers.BatchNormalization(momentum=0.9)(o)

        for _ in range(2):
            o = keras.layers.Conv2D(
                filters,
                (3, 3),
                padding="same",
                kernel_regularizer=keras.regularizers.l2(l=l2),
            )(o)
            o = tf.keras.layers.LeakyReLU(alpha=0.2)(o)

        o = keras.layers.MaxPool2D()(o)

        # NB: the d-SNE impl doesn't use dropout for the first conv block
        if dropout:
            o = keras.layers.Dropout(dropout)(o)

    model = keras.models.Model(inputs=[i], outputs=[o], name="embeds")
    return model


def dense_block(
    input_shape, dense_sizes=[1024, 128], l2=0.0001, batch_norm=False, dropout=0.5
):
    i = keras.layers.Input(shape=input_shape)
    o = keras.layers.Flatten()(i)

    dense_sizes = list(filter(bool, dense_sizes))

    for dense_size in dense_sizes:
        if dropout:
            o = keras.layers.Dropout(dropout)(o)
        o = keras.layers.Dense(
            dense_size,
            activation=None,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            name=f"dense_{dense_size}",
            kernel_regularizer=keras.regularizers.l2(l=l2),
        )(o)
        if batch_norm:
            o = keras.layers.BatchNormalization(momentum=0.9)(o)

        o = keras.layers.Activation("relu")(o)

    model = keras.models.Model(inputs=[i], outputs=[o], name="dense_layers")
    return model


def preds_block(
    input_shape,
    output_shape,
    l2=0.0001,
    dropout=0.5,
    l2_instance=False,
    angular_linear=False,
):
    i = keras.layers.Input(shape=input_shape)
    o = i

    # if dropout:
    #     o = keras.layers.Dropout(dropout)(o)

    if l2_instance:
        o = L2NormInstance()(o)

    if angular_linear:
        o = AngularLinear(output_shape, name="logits")(o)
    else:
        o = keras.layers.Dense(
            output_shape,
            activation=None,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            name="logits",
            kernel_regularizer=keras.regularizers.l2(l=l2),
        )(o)

    o = keras.layers.Activation("softmax", name="preds")(o)
    model = keras.models.Model(inputs=[i], outputs=[o], name="preds")
    return model


def logits_block(input_shape, dense_size, l2=0.0001, dropout=0.5):
    return keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(
                dense_size,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                name="logits",
                kernel_regularizer=keras.regularizers.l2(l=l2),
            ),
        ],
        name="logits",
    )


def model_attention(input_shape, embed_size, temperature=1.0):
    i = keras.layers.Input(shape=input_shape)
    iflat = keras.layers.Flatten()(i)
    inp = keras.layers.Lambda(lambda x: K.stop_gradient(x))(iflat)
    f = keras.layers.Dense(
        embed_size,
        activation=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        name="att_W1",
    )
    g = keras.layers.Dense(
        embed_size,
        activation=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        name="att_W2",
    )
    T = keras.layers.Lambda(K.transpose)
    dot = keras.layers.Lambda(lambda x: tf.tensordot(x[0], x[1], axes=(1)))
    temp = keras.layers.Lambda(
        lambda x: tf.scalar_mul(tf.constant(1.0 / temperature, dtype=DTYPE), x)
    )
    act = keras.layers.Activation("softmax", name="att_act")
    o = act(temp(dot([f(inp), T(g(inp))])))
    o2 = tf.add(o, T(o))
    model = keras.models.Model(inputs=[i], outputs=[o2])
    return model


def flip_elem(ds):
    for e in iter(ds):
        yield (e)
        ins, outs = e
        yield (
            (
                {
                    "input_source": ins["input_target"],
                    "input_target": ins["input_source"],
                },
                {
                    "preds": outs["preds_1"],
                    "preds_1": outs["preds"],
                    "aux_out": outs["aux_out"],
                },
            )
        )


def repeat_elem(ds, count=2):
    for e in iter(ds):
        for _ in range(count):
            yield (e)


def train(
    model,
    datasource,
    datasource_size,
    epochs,
    batch_size,
    callbacks,
    batch_repeats=1,
    flipping=False,
    checkpoints_path=None,
    verbose=1,
    val_datasource=None,
    val_datasource_size=None,
    val_freq=1,
    triangular_learning_rate=None,
):
    validation_steps = (
        ceil(val_datasource_size / batch_size) if val_datasource_size else None
    )
    steps_per_epoch = ceil(datasource_size / batch_size)

    if not val_datasource_size:
        val_datasource = None
        validation_steps = None

    if triangular_learning_rate:
        base_lr, max_lr = triangular_learning_rate / 4, triangular_learning_rate
        cyc_lr = CyclicLR(
            base_lr=base_lr,
            max_lr=max_lr,
            step_size=steps_per_epoch * epochs // 2,
            mode="triangular2",
        )
        callbacks = [cyc_lr, *callbacks]

    if flipping:
        datasource = flip_elem(datasource)
        steps_per_epoch *= 2

    if batch_repeats > 1:
        datasource = repeat_elem(datasource, batch_repeats)
        steps_per_epoch *= batch_repeats

    model.fit(
        datasource,
        validation_data=val_datasource,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=verbose,
        validation_freq=val_freq,
    )

    if checkpoints_path:
        if verbose:
            print("Restoring best checkpoint")
        model.load_weights(str(checkpoints_path.resolve()))


def train_flipping(  # noqa: C901
    model,
    datasource,
    datasource_size,
    epochs,
    batch_size,
    callbacks,
    checkpoints_path=None,
    verbose=1,
    val_datasource=None,
    val_datasource_size=None,
):
    if verbose:
        print("Train flipping")

    validation_steps = (
        ceil(val_datasource_size / batch_size) if val_datasource_size else None
    )
    steps_per_epoch = ceil(datasource_size / batch_size)

    if not val_datasource_size:
        val_datasource = None
        validation_steps = None

    train_iter = iter(datasource)
    val_loss_best = float("inf")

    for e in range(1, epochs + 1):
        if verbose:
            print("Epoch {}/{}.".format(e, epochs))

        for step in range(steps_per_epoch):
            ins, outs = next(train_iter)

            source_loss = model.train_on_batch(ins, outs)

            target_loss = model.train_on_batch(
                {
                    "input_source": ins["input_target"],
                    "input_target": ins["input_source"],
                },
                {
                    "preds": outs["preds_1"],
                    "preds_1": outs["preds"],
                    "aux_out": outs["aux_out"],
                },
            )

            if step % 10 == 0 and verbose:
                print(" Step {}/{}".format(step, steps_per_epoch))
                print(
                    "  Source Pass:  {}".format(
                        "  ".join(
                            [
                                "{} {:0.4f}".format(
                                    model.metrics_names[i], source_loss[i]
                                )
                                for i in range(len(source_loss))
                            ]
                        )
                    )
                )
                print(
                    "  Target Pass:  {}".format(
                        "  ".join(
                            [
                                "{} {:0.4f}".format(
                                    model.metrics_names[i], target_loss[i]
                                )
                                for i in range(len(target_loss))
                            ]
                        )
                    )
                )

        if validation_steps and val_datasource:
            val_iter = iter(val_datasource)
            val_loss = []
            for step in range(validation_steps):
                ins, outs = next(val_iter)
                val_loss.append(model.test_on_batch(ins, outs))

            val_loss_avg = reduce(
                lambda n, o: [n[i] + o[i] for i in range(len(val_loss))],
                val_loss,
                [0 for _ in val_loss[0]],
            )

            if verbose:
                print(
                    "  Validation:  {}".format(
                        "  ".join(
                            [
                                "{} {:0.4f}".format(
                                    model.metrics_names[i], val_loss_avg[i]
                                )
                                for i in range(len(val_loss_avg))
                            ]
                        )
                    )
                )

            if val_loss_avg[1] < val_loss_best:
                val_loss_best = val_loss_avg[1]
                if verbose:
                    print("val_preds_loss improved to {}".format(val_loss_best))
                if checkpoints_path:
                    model.save(str(checkpoints_path.resolve()))

    if checkpoints_path:
        if verbose:
            print("Restoring best checkpoint")
        model.load_weights(str(checkpoints_path.resolve()))


def recompile(model, architecture="single_stream"):
    if architecture == "single_stream":
        metric = K.get_value(model.metrics)
    else:
        metric = {
            "preds": "accuracy",
            "preds_1": "accuracy",
        }  # there seems to be an issue with K.get_value(model.metrics) in case of two-stream architectures

    model.compile(
        loss=K.get_value(model.loss) if hasattr(model, "loss") else None,
        loss_weights=K.get_value(model.loss_weights)
        if hasattr(model, "loss_weights")
        else None,
        optimizer=K.get_value(model.optimizer) if hasattr(model, "optimizer") else None,
        metrics=metric,
    )


def train_gradual_unfreeze(
    model,
    model_base_name,
    datasource,
    datasource_size,
    epochs,
    batch_size,
    callbacks,
    architecture="single_stream",
    checkpoints_path=None,
    triangular_learning_rate: float = None,
    lr_fact=0.25,
    verbose=1,
    val_datasource=None,
    val_datasource_size=None,
):
    validation_steps = (
        ceil(val_datasource_size / batch_size) if val_datasource_size else None
    )
    steps_per_epoch = ceil(datasource_size / batch_size)

    if not val_datasource_size:
        val_datasource = None
        validation_steps = None

    model_base = model.get_layer(model_base_name)

    max_unfreeze, step_size = {
        "vgg16": (12, 4),
        "resnet50": (20, 5),
        "resnet50v2": (20, 5),
        "resnet101v2": (24, 6),
        "resnet152v2": (24, 6),
    }[model_base_name]

    if triangular_learning_rate:
        base_lr, max_lr = triangular_learning_rate / 4, triangular_learning_rate

    for num_unfreeze in range(step_size, max_unfreeze + 1, step_size):
        freeze(model_base, num_unfreeze)
        recompile(model, architecture)

        if verbose:
            print("Training with {} base_model layers unfrozen.".format(num_unfreeze))
            model.summary()

        if triangular_learning_rate:
            cyc_lr = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size=steps_per_epoch * epochs // 2, mode="triangular2")  # type: ignore
            base_lr, max_lr = lr_fact * base_lr, lr_fact * max_lr  # type: ignore
            callbacks = [cyc_lr, *callbacks]

        model.fit(
            datasource,
            validation_data=val_datasource,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=verbose,
        )

        if checkpoints_path:
            model.load_weights(str(checkpoints_path.resolve()))
