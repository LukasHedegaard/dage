import tensorflow as tf
keras = tf.compat.v2.keras

class TerminateOnNegMetric(keras.callbacks.Callback):
    def __init__(self, metric='aux_loss'):
        super(TerminateOnNegMetric, self).__init__()
        self.metric = metric

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        metric = logs.get(self.metric)
        if metric is not None:
            if metric < -0.1:
                print('Batch %d: Invalid metric, terminating training\n' % (batch))
                self.model.stop_training = True

def checkpoint(checkpoints_dir, monitor='loss',verbose=True):
    return keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoints_dir / 'cp-best.ckpt'),
        monitor=monitor,
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        period=1,  
        verbose=verbose
    )

def reduce_lr(monitor='loss', verbose=True):
    return keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=0.1,
        patience=3,
        min_lr=1e-6,
        verbose=verbose
    )


def early_stop(monitor='loss', verbose=True):
    return keras.callbacks.EarlyStopping(
        monitor=monitor, 
        min_delta=0, 
        patience=7, 
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
        tensorboard(tensorboard_dir),
        TerminateOnNegMetric(),
    ]