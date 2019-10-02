import tensorflow as tf
import numpy as np
from PIL import Image

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_image(image_path, shape):
    image = Image.open(image_path)
    image = image.resize(shape[:2])
    image = np.array(image).tobytes()
    return image 


def serialise_image(image, label):
    feature = {'label': _int64_feature(label),
               'image': _bytes_feature(tf.compat.as_bytes(image))}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def deserialise_image(serialised_example, shape):
    feature = {'image': tf.FixedLenFeature([], tf.string, default_value=''),
               'label': tf.FixedLenFeature([], tf.int64, default_value=0)}

    parsed_example = tf.io.parse_single_example(
        serialized=serialised_example,
        features=feature
    )
    image = tf.decode_raw(parsed_example['image'], tf.uint8)
    image = tf.reshape(image, list(shape))
    parsed_example['image'] = image #preprocess_image(image, shape) 
    return parsed_example


def preprocess_image(uint8image, shape):
    # assuming shape is (*, *, 3)
    channel_shape = [*shape[:2], 1]
    imagenet_avg = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225] #tf.constant([0.229, 0.224, 0.225])
    avg = tf.concat([tf.fill(channel_shape, imgn_a) for imgn_a in imagenet_avg], axis=2)
    std = tf.concat([tf.fill(channel_shape, imgn_s) for imgn_s in imagenet_std], axis=2)
    
    float_image = tf.cast(uint8image, tf.float32)
    float_image.set_shape(list(shape))
    float_image = float_image / 255.0
    float_image = float_image - avg
    float_image = float_image / std
    return float_image #tf.cast(float_image, tf.uint8)


def open_writer(file_path):
    opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    return tf.python_io.TFRecordWriter(file_path, options=opts)


def load_dataset(file_names, data_shape):
    dataset = tf.data.TFRecordDataset(file_names, compression_type='GZIP')
    dataset = dataset.map(lambda i: deserialise_image(i, data_shape))
    dataset = dataset.map(lambda ex: (ex['image'], ex['label']))
    return dataset
