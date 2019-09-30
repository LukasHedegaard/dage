import os
import json
import shutil

import numpy as np
import argparse as ap
import tensorflow as tf
import tfrecord_util as tfr

from os import path
from glob import glob
from pathlib import Path
from pprint import pprint
from datetime import datetime
from importlib import import_module
from sklearn.metrics import classification_report

MODEL_CONFIG_DIR = 'model_configs'
MODEL_CONFIG_ABS_DIR = path.join(path.dirname(
    path.abspath(__file__)), MODEL_CONFIG_DIR)
MODEL_CONFIGS = [path.splitext(f)[0]
                 for f in os.listdir(MODEL_CONFIG_ABS_DIR)
                 if (path.isfile(path.join(MODEL_CONFIG_ABS_DIR, f)) and '__init__' not in f)
                 ]
DATASETS = {
    "office31": ["amazon", "dslr", "webcam"]
}


def parse_args():
    parser = ap.ArgumentParser(description='Train a model.')
    parser.add_argument('--model-config', '-m',
                        type=check_config,
                        required=True,
                        help='The model configuration to train.')
    parser.add_argument('--source-dataset-name', '-sd',
                        type=check_dataset_name,
                        required=False,
                        default='office31/amazon',
                        help='The source dataset name.')
    # parser.add_argument('--target-dataset', '-sd',
    #                     type=check_dataset_name,
    #                     required=False,
    #                     default='office31/dslr',
    #                     help='The target dataset name.')
    parser.add_argument('--data-dir', '-d',
                        type=check_data_dir,
                        required=False,
                        default='data',
                        help='The directory where to find the prepared data.')
    parser.add_argument('--run-id',
                        type=str,
                        required=False,
                        default=None,
                        help='Experiment ID. Used to track build number in Jenkins.')
    parser.add_argument('--run-info',
                        type=str,
                        required=False,
                        default=None,
                        help='Experiment information. Used to track build number in Jenkins.')
    parser.add_argument('--output-dir', '-o',
                        type=str,
                        required=False,
                        default='runs',
                        help='The directory where to store the artifacts of the training.')
    parser.add_argument('--epochs', '-e',
                        type=int,
                        required=False,
                        default=10,
                        help='Number of epochs. Default: 10')
    parser.add_argument('--batch-size', '-b',
                        type=int,
                        required=False,
                        default=16,
                        help='Batch size. Default: 16')
    parser.add_argument('--optimiser',
                        type=str,
                        required=False,
                        default='momentum',
                        help='The optimiser use for training the model. Default: momentum')
    parser.add_argument('--learning-rate', '-lr',
                        type=float,
                        required=False,
                        default=0.01,
                        help='Learning rate. Default: 0.01')
    parser.add_argument('--shuffle-buffer-size', '-sb',
                        type=int,
                        required=False,
                        default=1048,
                        help='Shuffle buffer size. Default: 1048')
    parser.add_argument('--weights-path', '-w',
                        type=str,
                        required=False,
                        default=None,
                        help='The location of a pre-trained weights file. Default: None')
    parser.add_argument('--early-stopping-patience',
                        type=int,
                        required=False,
                        default=30,
                        help='Early stoppping patience. Default: 30')
    parser.add_argument('--lr-reduction-patience',
                        type=int,
                        required=False,
                        default=10,
                        help='Learning rate reduction on plateau patience. Default: 10')
    return parser.parse_args()


def check_config(config_name):
    config_name = path.basename(config_name)
    config_name = config_name.replace('.py', '')
    if config_name not in MODEL_CONFIGS:
        msg = 'Model configuration "{0}" is unknown!'.format(config_name)
        raise ap.ArgumentTypeError(msg)
    return config_name


def check_data_dir(dir_name):
    if not path.exists(dir_name):
        msg = 'Data directory "{0}" does not exist!'.format(dir_name)
        raise ap.ArgumentTypeError(msg)
    return dir_name


def check_dataset_name(dataset_name):
    parts = dataset_name.split('/') # expecting e.g. office31/amazon
    if parts[0] not in DATASETS.keys():
        msg = 'Dataset "{0}" is not an available dataset. Choose from {1}'.format(dataset_name, list(DATASETS.keys()))
        raise ap.ArgumentTypeError(msg)
    if parts[1] not in DATASETS[parts[0]]:
        msg = 'Sub-dataset "{0}" is not an available dataset. Choose from {1}'.format(dataset_name, DATASETS[parts[0]])
        raise ap.ArgumentTypeError(msg)
    return dataset_name


# represents summary of data (generated when generating tf records)
class DataSummary:
    def __init__(self, file_path: Path):
        print('Loading summary from {0}'.format(file_path))
        with file_path.open('r') as f:
            content = json.load(f)
        self.classes = content['classes']
        self.n_classes = len(self.classes)
        self.data_shape = (content['data_shape']['width'], content['data_shape']['height'], content['data_shape']['depth'])
        self.train_size = content['train']['num_samples']
        self.validation_size = content['validation']['num_samples']
        self.test_size = content['test']['num_samples']


class ModelTrainer:
    def __init__(self, output_dir, source_dataset_name, data_dir, model_config, batch_size, shuffle_buffer_size,
                 epochs, optimiser, learning_rate, weights_path, run_id, run_info,
                 early_stopping_patience, lr_reduction_patience):
        self.output_dir = Path(output_dir)
        self.source_dataset_name = source_dataset_name
        self.data_dir = Path(data_dir)
        self.model_config = model_config
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.optimiser = optimiser
        self.learning_rate = learning_rate
        self.restore_weights_path = weights_path
        self.run_info = run_info
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.lr_reduction_patience = lr_reduction_patience
        self.verbose = 1

        data_summary_path = self.data_dir / self.source_dataset_name / 'summary.json'
        self.data_summary = DataSummary(data_summary_path)
        self.full_model_name = '{}_{}'.format(
            self.model_config,
            self.source_dataset_name.replace('/','_')
        )

        if run_id is None or len(str(run_id).strip()) == 0:
            self.run_id = datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
        else:
            self.run_id = str(run_id).strip()

        self.run_dir = Path(output_dir) / '{}_{}'.format(self.full_model_name, self.run_id)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Copy the summary
        shutil.copy2(str(data_summary_path), str(self.run_dir / 'data_summary.json'))

        self.checkpoints_dir = self.run_dir / 'weights'
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.tensorboard_dir = self.run_dir / 'logs'
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        self.model_arch_path = str(self.run_dir / 'architecture.json')
        self.prediction_file_path = str(self.run_dir / 'predictions.csv')
        self.report_path = str(self.run_dir / 'report.txt')

        self.model = None

        self._save_train_params(str(self.run_dir / 'train_params.json'))


    def _weigths_path_exists(self):
        return self.restore_weights_path is not None and path.exists(self.restore_weights_path)


    def _save_train_params(self, train_params_path: str):
        data = {
            'run_id': self.run_id,
            'run_info': self.run_info,
            'data_dir': str(self.data_dir),
            'source_dataset_name': self.source_dataset_name,
            'batch_size': self.batch_size,
            'shuffle_buffer_size': self.shuffle_buffer_size,
            'optimiser': self.optimiser,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'lr_reduction_patience': self.lr_reduction_patience,
            'restore_weights_path': self.restore_weights_path if self._weigths_path_exists() else '',
        }
        with open(train_params_path, 'w') as file:
            json.dump(data, file, indent=2)


    def _load_dataset(self, split_name):
        file_pattern = str(self.data_dir / self.source_dataset_name / split_name / '*.tfrecords')
        file_names = glob(file_pattern)
        dataset = tf.data.TFRecordDataset(file_names, compression_type='GZIP')
        dataset = dataset.map(lambda i: tfr.deserialise_image(i, self.data_summary.data_shape))
        dataset = dataset.map(lambda ex: (ex['image'], ex['label']))
        return dataset


    def _generate_prediction_file(self, model):
        print('Creating predictions file {}...'.format(self.prediction_file_path))
        test_size = self.data_summary.test_size
        test_steps = int(np.ceil(test_size / self.batch_size))
        test_set = self._load_dataset('test').batch(self.batch_size)
        test_iter = test_set.make_one_shot_iterator()
        tf_session = tf.keras.backend.get_session()
        y_true = []
        y_pred = []
        top_k = 10
        with open(self.prediction_file_path, 'w') as f:
            next_element = test_iter.get_next()
            for _ in range(test_steps):
                result = tf_session.run([next_element])
                docs, true_labels, doc_ids = result[0]
                predicted_proba = model.predict(docs, steps=1, verbose=0)
                predicted_labels = predicted_proba.argmax(axis=1)
                for j in range(len(doc_ids)):
                    f.write('{},{}'.format(doc_ids[j], true_labels[j]))
                    sorted_label_idx = predicted_proba[j].argsort(
                    )[-top_k:][::-1]
                    for k_label in sorted_label_idx:
                        k_proba = predicted_proba[j][k_label]
                        f.write(',{},{:.5f}'.format(k_label, k_proba))
                    f.write('\n')
                    y_true.append(true_labels[j])
                    y_pred.append(predicted_labels[j])

        # Generate classification report
        with open(self.report_path, 'w') as file:
            file.write('# Classification Report\n')
            file.write(classification_report(y_true=y_true, y_pred=y_pred))


    def _build_fit_callbacks(self):
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            str(self.checkpoints_dir / 'checkpoint.hdf5'),
            monitor='val_acc',
            save_best_only=True,
            save_weights_only=True,
            mode='auto',
            period=5,  # Save weights, every 5 epoch.
            verbose=self.verbose
        )
        stop_early = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=self.early_stopping_patience,
            mode='auto',
            verbose=self.verbose
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=self.lr_reduction_patience,
            min_lr=1e-5,
            verbose=self.verbose
        )
        tb = tf.keras.callbacks.TensorBoard(log_dir=self.tensorboard_dir)
        return [checkpoint, stop_early, reduce_lr, tb]


    def run(self):
        print('Starting the training on {}'.format(self.source_dataset_name))

        train_size = self.data_summary.train_size
        validation_size = self.data_summary.validation_size
        test_size = self.data_summary.test_size

        train_steps = int(np.ceil(train_size / self.batch_size))
        validation_steps = int(np.ceil(validation_size / self.batch_size))
        test_steps = int(np.ceil(test_size / self.batch_size))

        print('Training set size:   {0} ({1} steps)'.format(train_size, train_steps))
        print('Validation set size: {0} ({1} steps)'.format(validation_size, validation_steps))
        print('Test set size:       {0} ({1} steps)'.format(test_size, test_steps))

        mm = import_module('{0}.{1}'.format(MODEL_CONFIG_DIR, self.model_config))
        print('Building model configuration {0}'.format(mm.__name__))

        model = mm.build(
            input_shape=self.data_summary.data_shape,
            n_classes=self.data_summary.n_classes,
        )

        if self.optimiser == 'adam':
            opt = tf.keras.optimizers.Adam(lr=self.learning_rate)
        elif self.optimiser == 'sgd':
            opt = tf.keras.optimizers.SGD(lr=self.learning_rate)
        else:
            opt = tf.keras.optimizers.SGD(lr=self.learning_rate, momentum=0.9)
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )
        print(model.summary())

        model_json = model.to_json(indent=2)
        with open(self.model_arch_path, 'w') as json_file:
            json_file.write(model_json)

        if self._weigths_path_exists():
            latest = tf.train.latest_checkpoint(self.restore_weights_path)
            if latest is not None:
                print('Loading latest checkpoint weights from {0}'.format(latest))
                model.load_weights(latest)
            print('Restoring weights from {}'.format(self.restore_weights_path))
        else:
            print('Weights path "{}" not found!'.format(self.restore_weights_path))

        training_set = self._load_dataset('train').take(self.batch_size * train_steps).repeat().shuffle(self.shuffle_buffer_size).batch(self.batch_size)
        validation_set = self._load_dataset('validation').take(self.batch_size * validation_steps).repeat().batch(self.batch_size)
        test_set = self._load_dataset('test').batch(self.batch_size)

        model.fit(
            training_set,
            epochs=self.epochs,
            steps_per_epoch=train_steps,
            validation_data=validation_set,
            validation_steps=validation_steps,
            callbacks=self._build_fit_callbacks(),
            verbose=self.verbose,
        )
        results = model.evaluate(test_set, steps=test_steps)
        print('test loss {:0.4f},  test acc: {:0.4f}'.format(
            results[0], results[1]))

        self._generate_prediction_file(model)


def main():
    args = vars(parse_args())
    print('Training a configuration using following parameters: {0}'.format(args))
    md = ModelTrainer(**args)
    md.run()


if __name__ == '__main__':
    main()
