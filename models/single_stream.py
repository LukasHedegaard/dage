import tensorflow as tf

from models.common import dense_block, freeze, get_output_shape, preds_block

keras = tf.compat.v2.keras


def model(
    model_base,
    input_shape,  # currently unused
    output_shape,
    optimizer,
    num_unfrozen_base_layers=0,
    embed_size=128,
    dense_size=1024,
    l2=0.0001,
    batch_norm=True,
    dropout=0.5,
):
    freeze(model_base, num_leave_unfrozen=num_unfrozen_base_layers)

    model_mid = dense_block(
        input_shape=get_output_shape(model_base),
        dense_sizes=[dense_size, embed_size],
        l2=l2,
        batch_norm=batch_norm,
        dropout=dropout,
    )
    model_top = preds_block(
        input_shape=get_output_shape(model_mid), output_shape=output_shape, l2=l2
    )

    model = keras.Sequential([model_base, model_mid, model_top])
    model_features = keras.Sequential([model_base, model_mid])

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        loss_weights=None,
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    return model, model, model_features
