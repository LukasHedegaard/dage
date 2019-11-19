from pathlib import Path
import argparse
import numpy as np
import tensorflow as tf
keras = tf.compat.v2.keras
Dataset = tf.compat.v2.data.Dataset
AUTOTUNE = tf.data.experimental.AUTOTUNE
DTYPE = tf.float32
from scipy.io import savemat
from functools import partial
from gpu import setup_gpu
import dataset_gen as dsg

def parse_args():
    parser = argparse.ArgumentParser(description='Generate tfrecord features from dataset')

    # parser.add_argument('--dataset', type=str, default='Office31', help='Dataset to generate. Default: Office31')
    # parser.add_argument('--source_dir', type=str, default='datasets', help='Directory of dataset to generate features from. Default: datasets')
    # parser.add_argument('--sink_dir', type=str, default='datasets/features', help='Directory in which to put features. Default: datasets/features')
    parser.add_argument('--feature_extractor', type=str, default='vgg16', help='Which feature-extraction method to use. Default: vgg16')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU id to use. Default: 0')
    parser.add_argument('--batch_size', type=int, default='64', help='Batch size. Default: 64')

    return parser.parse_args()


def main(args):
    if args.gpu_id:
        setup_gpu(args.gpu_id)

    # Prep feature-extractor
    INPUT_SHAPE = (224, 224, 3)

    preprocess_input = {
        'vgg16'      : partial(keras.applications.vgg16.preprocess_input, mode='tf'),
        'resnet101v2': partial(keras.applications.resnet_v2.preprocess_input, mode='tf'), #NB: tf v 1.15 has a minor bug in keras_applications.resnet. Fix: change the function signature to "def preprocess_input(x, **kwargs):""
    }[args.feature_extractor] or None

    feature_extractor = {
        'vgg16'      : lambda: keras.applications.vgg16.VGG16 (input_shape=INPUT_SHAPE, include_top=False, weights='imagenet'),
        'resnet101v2': lambda: keras.applications.resnet_v2.ResNet101V2(input_shape=INPUT_SHAPE, include_top=False, weights='imagenet'),
    }[args.feature_extractor]()
    feature_extractor.summary()

    # Prep source dataset (Office31 is currently the only option)
    project_base_path = Path(__file__).parent.parent

    for domain in ['amazon','webcam','dslr']:
        ds_path = project_base_path / 'datasets' / 'Office31' / domain / 'images'
        data_paths = [str(p) for p in ds_path.glob('*/*.jpg')]

        prep_image = dsg.make_image_prep(INPUT_SHAPE[:2], preprocess_input)
    
        ds = Dataset.list_files(data_paths, shuffle=False) \
            .map(prep_image, num_parallel_calls=AUTOTUNE) \
            .batch(int(args.batch_size)) \
            .prefetch(buffer_size=AUTOTUNE)

        # Perform feature-extraction
        features = feature_extractor.predict_generator(ds, verbose=1)

        for feature, data_path in zip(features, data_paths):
            p = data_path.replace('images',args.feature_extractor).replace('.jpg','.mat')
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            savemat(p, {'data': feature})
        

if __name__ == '__main__':
    args = parse_args()
    main(args)