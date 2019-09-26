import tensorflow as tf
import numpy as np


def wrap_int64(value):
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def wrap_bytes(value):
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def wrap_float(value):
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def serialise_bow(document_id, document_length, bow_count, bow_tfidf,
                  vocab_size, level2_label, level3_label):
    """
    Creates a `tf.train.SequenceExample` for a given sample in Bag-of-Words representation.
    :param document_id: the document id
    :param document_length: the number of tokens/words in the document
    :param bow_count: the bag-of-words representation of the document based on counts
    :param bow_tfidf: the bag-of-words representation of the document based on TF-IDF
    :param vocab_size: the size of the vocabulary
    :param level2_label: level 2 label
    :param level3_label: level 3 label
    :return:
    """
    count_indices_1, count_values = zip(*bow_count)
    count_indices_0 = [0] * len(count_indices_1)
    tfidf_indices_1, tfidf_values = zip(*bow_tfidf)
    tfidf_indices_0 = [0] * len(tfidf_indices_1)

    example = tf.train.SequenceExample(
        context=tf.train.Features(feature={
            'id': wrap_int64(document_id),
            'doc_tfidf_indices_0': wrap_int64(tfidf_indices_0),
            'doc_tfidf_indices_1': wrap_int64(tfidf_indices_1),
            'doc_tfidf_values': wrap_float(tfidf_values),
            'doc_count_indices_0': wrap_int64(count_indices_0),
            'doc_count_indices_1': wrap_int64(count_indices_1),
            'doc_count_values': wrap_int64(count_values),
            'doc_length': wrap_int64(document_length),
            'l2_label': wrap_int64(level2_label),
            'l3_label': wrap_int64(level3_label),
        })
    )
    return example.SerializeToString()


def decode_sparse_tensor(parsed_example, key):
    doc = parsed_example[key]
    doc = tf.sparse.reshape(doc, (-1,))
    parsed_example[key] = doc


def deserialise_bow(serialised_example, bow_size):
    features = {
        'id': tf.FixedLenFeature([], tf.int64),
        'doc_tfidf': tf.SparseFeature(
            index_key=['doc_tfidf_indices_0', 'doc_tfidf_indices_1'],
            value_key='doc_tfidf_values',
            dtype=tf.float32,
            size=[1, bow_size]
        ),
        'doc_count': tf.SparseFeature(
            index_key=['doc_count_indices_0', 'doc_count_indices_1'],
            value_key='doc_count_values',
            dtype=tf.int64,
            size=[1, bow_size]
        ),
        'doc_length': tf.FixedLenFeature([], tf.int64),
        'l2_label': tf.FixedLenFeature([], tf.int64),
        'l3_label': tf.FixedLenFeature([], tf.int64),
    }

    parsed_example = tf.io.parse_single_example(
        serialized=serialised_example,
        features=features
    )

    decode_sparse_tensor(parsed_example, 'doc_count')
    decode_sparse_tensor(parsed_example, 'doc_tfidf')

    parsed_example['doc_count'] = tf.cast(parsed_example['doc_count'], tf.int32)

    return parsed_example


def serialise_document(document_id, document_length, document, level2_label, level3_label):
    example = tf.train.SequenceExample(
        context=tf.train.Features(feature={
            'id': wrap_int64(document_id),
            'doc': wrap_int64(document),
            'doc_length': wrap_int64(document_length),
            'l2_label': wrap_int64(level2_label),
            'l3_label': wrap_int64(level3_label),
        })
    )
    return example.SerializeToString()


def deserialise_document(serialised_example, fixed_length=None):
    features = {
        'id': tf.FixedLenFeature([], tf.int64),
        'doc': tf.VarLenFeature(tf.int64),
        'doc_length': tf.FixedLenFeature([], tf.int64),
        'l2_label': tf.FixedLenFeature([], tf.int64),
        'l3_label': tf.FixedLenFeature([], tf.int64),
    }
    parsed_example = tf.parse_single_example(
        serialized=serialised_example,
        features=features
    )
    doc = tf.sparse.to_dense(parsed_example['doc'])
    if isinstance(fixed_length, int) and fixed_length > 0:
        doc = convert_to_fixed_length_tensor(doc, fixed_length)

    parsed_example['doc'] = doc
    return parsed_example


def convert_to_fixed_length_tensor(doc_tensor, length, pad_value=0):
    """Takes a document tensor of variable length and makes it fixed-length tensor"""
    t_fixed_length = tf.constant(length, tf.int64)

    doc_length = tf.dtypes.cast(tf.shape(doc_tensor)[0], dtype=tf.int64)

    # Pad tensor if its size is less than required length
    diff = tf.subtract(t_fixed_length, doc_length)
    padding_diff = tf.maximum(diff, 0)
    doc_tensor = tf.pad(doc_tensor, [[0, padding_diff]], constant_values=pad_value)

    # Truncate tensor if tensor length is larger than required length
    doc_length = tf.dtypes.cast(tf.shape(doc_tensor)[0], dtype=tf.int64)
    to_index = tf.minimum(t_fixed_length, doc_length)
    doc_tensor = tf.slice(doc_tensor, [0], [to_index])

    return doc_tensor


def open_writer(file_path):
    opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    return tf.python_io.TFRecordWriter(file_path, options=opts)
