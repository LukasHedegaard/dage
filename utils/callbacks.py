import tensorflow as tf
keras = tf.compat.v2.keras

def checkpoint(checkpoints_dir, monitor='loss',verbose=True):
    return keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoints_dir / 'cp-{epoch:003d}.ckpt'),
        monitor=monitor,
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        period=5,  # Save weights, every 5 epoch.
        verbose=verbose
    )

def reduce_lr(monitor='loss', verbose=True):
    return keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=0.1,
        patience=5,
        min_lr=1e-5,
        verbose=verbose
    )


def early_stop(monitor='loss', verbose=True):
    return keras.callbacks.EarlyStopping(
        monitor=monitor, 
        min_delta=0, 
        patience=10, 
        verbose=verbose, 
        mode='auto', 
        baseline=None, 
        restore_best_weights=True
    )

def tensorboard(tensorboard_dir):
    return keras.callbacks.TensorBoard(log_dir=tensorboard_dir)


def all(checkpoints_dir, tensorboard_dir, monitor='loss', verbose=True):
    return [
        checkpoint(checkpoints_dir, monitor, verbose), 
        reduce_lr(monitor, verbose), 
        early_stop(monitor, verbose), 
        tensorboard(tensorboard_dir)
    ]