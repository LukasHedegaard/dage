import tensorflow as tf
import numpy as np


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


#was generate_tf_example
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
    parsed_example['image'] = tf.reshape(image, list(shape))
    return parsed_example


def open_writer(file_path):
    opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    return tf.python_io.TFRecordWriter(file_path, options=opts)
