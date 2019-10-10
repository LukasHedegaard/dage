import os
import numpy as np
import argparse as ap

from os import path
from pprint import pprint
from typing import List, Union
from importlib import import_module
from time import time
from pathlib import Path

import tensorflow as tf


class EarlyStopping:
    def __init__(self, patience: int = 10):
        self.patience = patience
        self.best = np.Inf
        self.wait = 0

    def on_train_begin(self):
        self.best = np.Inf
        self.wait = 0

    def reached(self, epoch: int, current: float):
        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
        should_stop = self.patience > 0 and self.wait >= self.patience
        tf.print('Early stopping evaluation. Epoch={} best={} wait={} current={} stop={}'.format(
            epoch, self.best, self.wait, current, should_stop
        ))
        return should_stop


class ModelCheckpoint:
    def __init__(self, model, checkpoint_dir: Path, pretrained_checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.pretrained_checkpoint_dir = pretrained_checkpoint_dir
        self.best_loss = np.Inf
        self.checkpoint = tf.compat.v2.train.Checkpoint(model=model)
        self.ckpt_manager = tf.train.CheckpointManager(self.checkpoint, str(checkpoint_dir), max_to_keep=1)

    def restore_if_any(self):
        pretrained_weights_path = self._get_pretrained_weights()
        if pretrained_weights_path is not None:
            tf.print('Restoring pre-trained weights from {}'.format(pretrained_weights_path))
            self.checkpoint.restore(str(pretrained_weights_path))
        else:
            tf.print('Not using pre-trained weights.')

    def save_best(self, current_loss: float):
        if current_loss < self.best_loss:
            tf.print('Loss improved form {:0.4f} to {:0.4f}. Saving checkpoint to {}'.format(
                self.best_loss, current_loss, self.checkpoint_dir
            ))
            self.best_loss = current_loss
            self.ckpt_manager.save()

    def _get_pretrained_weights(self):
        mgr = tf.train.CheckpointManager(self.checkpoint, str(self.pretrained_checkpoint_dir), max_to_keep=1)
        return mgr.latest_checkpoint


class ModelTrainer:
    def __init__(self,
        model,
        learning_rate: float,
        output_dir: Path,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.output_dir = output_dir
        self._early_stopping = EarlyStopping()

    def run(self):
        # Import the module containing the model
        mm = import_module('{0}.{1}'.format(path.basename(MODEL_CONFIG_DIR), self.model_config))

        # Create new instance of the model
        input_specs = self.data_spec.get_input_specs()
        n_classes = self.data_spec.n_classes
        model = mm.new(input_specs, n_classes, self.data_spec.target_name)

        tf.print('The model contains {} tf.Variable:'.format(len(model.trainable_variables)))
        n_vars = 0
        for v in model.trainable_variables:
            tf.print(' - name={}  shape={}'.format(v.name, v.shape))
            n_vars += np.prod(v.shape.as_list())

        tf.print('Total number of variables: {:,}'.format(n_vars))

        self.train(model)
        self.test(model)

        print('Training completed successfully.')

    def train(self, model):
        # Prepare training set
        train_files, validation_files = self.data_spec.split_train()
        training_set = self.data_spec.create_dataset(train_files, self.shuffle_buffer_size)
        training_set = training_set.batch(self.batch_size)
        training_set = training_set.repeat(1)

        validation_set = self.data_spec.create_dataset(validation_files, self.shuffle_buffer_size)
        validation_set = validation_set.batch(self.batch_size)
        validation_set = validation_set.repeat(1)

        optimizer = tf.compat.v2.optimizers.Adam(learning_rate=self.learning_rate)

        checkpoint = ModelCheckpoint(optimizer, model, Path('./runs/chkpt'), './runs/chkpt')
        checkpoint.restore_if_any()

        tf.print('Beginning training')

        self._early_stopping.on_train_begin()

        for epoch in range(self.epochs):
            train_loss = tf.compat.v2.metrics.Mean(name='train_loss')
            train_accuracy = tf.compat.v2.metrics.CategoricalAccuracy(name='train_accuracy')

            train_started_at = time()

            for step, (x, y) in enumerate(training_set):
                loss, y_hat = ModelTrainer._train_step(epoch, step, model, x, y, optimizer)

                train_loss(loss)
                train_accuracy(y, y_hat)

                if step % 50 == 0:
                    tf.print('  Step {:4d}: acc {:0.4f}  loss {}'.format(
                        step,
                        train_accuracy.result(),
                        train_loss.result()
                    ))

            train_time = time() - train_started_at

            val_loss = tf.compat.v2.metrics.Mean(name='validation_loss')
            val_acc = tf.compat.v2.metrics.CategoricalAccuracy(name='validation_accuracy')
            for step, (x, y) in enumerate(validation_set):
                logits = model.call(x, training=False)
                loss = model.compute_loss(y, logits)
                y_hat = tf.nn.softmax(logits)
                val_acc(y, y_hat)
                val_loss(loss)

            tf.print(
                'epoch: {:3d}  train_acc: {:0.4f}  val_acc: {:0.4f}  train_loss: {} val_loss: {} train time: {:0.0f} sec'.format(
                    epoch + 1,
                    train_accuracy.result(),
                    val_acc.result(),
                    train_loss.result(),
                    val_loss.result(),
                    train_time
                ))

            checkpoint.save_best(val_loss.result())

            if self._early_stopping.reached(epoch, val_loss.result()):
                tf.print('Early stopping criteria reached. Stopping...')
                break

    @staticmethod
    def _train_step(epoch, step, model, x, y, optimizer):
        with tf.GradientTape() as tape:
            logits = model.call(x, training=True)
            loss = model.compute_loss(y, logits)

        # Compute and apply gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        y_hat = tf.nn.softmax(logits)

        return loss, y_hat

    def test(self, model):
        test_set = self.data_spec.create_dataset(self.data_spec.test_file_names, self.shuffle_buffer_size)
        test_set = test_set.batch(self.batch_size)
        test_set = test_set.repeat(1)

        test_loss = tf.compat.v2.metrics.Mean(name='test_loss')
        test_accuracy = tf.compat.v2.metrics.CategoricalAccuracy(name='test_accuracy')

        tf.print('Predicting on the test set...')

        started_at = time()

        for step, (x, y) in enumerate(test_set):
            logits = model.call(x, training=False)
            loss = model.compute_loss(y, logits)
            y_hat = tf.nn.softmax(logits)
            test_accuracy(y, y_hat)
            test_loss(loss)

        tf.print('Test acc: {:0.4f}  loss: {}  time: {:0.0f} sec'.format(
            test_accuracy.result(),
            test_loss.result(),
            time() - started_at
        ))


