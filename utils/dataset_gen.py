from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from pathlib import Path
from typing import List, Union, Optional, Dict, Tuple, Callable
import random
from functools import lru_cache, partial
import itertools
from utils.file_io import load_json
import tensorflow as tf
# import tensorflow_addons as tfa
from math import pi as PI
from scipy.io import loadmat
AUTOTUNE = tf.data.experimental.AUTOTUNE
Dataset = tf.compat.v2.data.Dataset
DTYPE = tf.float32

def make_image_prep(
    shape=[224,224,3], 
    preprocess_input_fn:Callable=None, 
):
    shape = shape[:2]
    def prep(file_path):
        if isinstance(file_path, str):
            file_path = tf.constant(file_path)
        img = tf.compat.v2.io.read_file(file_path) # load the raw data from the file as a string
        img = tf.compat.v2.image.decode_jpeg(img, channels=3) # convert the compressed string to a 3D uint8 tensor
        img = tf.compat.v2.image.convert_image_dtype(img, DTYPE) # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        if preprocess_input_fn:
            img = preprocess_input_fn(img*255) # preprocess input assumes input in [0,255] range.
        img = tf.compat.v2.image.resize(img, shape) # resize the image to the desired size.
        return img
    return prep


def make_mat_prep(
    shape=[224,224,3], 
    preprocess_input_fn:Callable=None, 
):
    def prep(file_path):
        d = loadmat(file_path)
        if preprocess_input_fn:
            d = preprocess_input_fn(d) # should pick the right column from the dict above
        return d
    return prep


def dataset_from_paths(
    data_paths: List[str],
    preprocess_input:Callable=None,
    shape=[224,224,3], 
    seed=1,
    shuffle=True,
):
    CLASS_NAMES = tf.constant(sorted(list(set([p.split('/')[-2] for p in data_paths]))))
    data_type = Path(data_paths[0]).suffix if data_paths else 'none'

    prep_dat = {
        # '.jpg': lambda p, fn=make_image_prep(shape, preprocess_input): fn(tf.constant(p, dtype=tf.string)),
        '.jpg': make_image_prep(shape, preprocess_input),
        '.mat': make_mat_prep(shape, preprocess_input),
        'none': lambda x: x
    }[data_type]

    def prep_label(file_path):
        fp = tf.constant(file_path)
        parts = tf.compat.v2.strings.split(fp, '/') # convert the path to a list of path components
        label = tf.equal(CLASS_NAMES, parts[-2]) # The second to last is the class-directory
        return label
        
    def gen():
        for fp in data_paths:
            yield prep_dat(fp), prep_label(fp)

    output_types = (DTYPE, tf.bool)
    output_shapes = (tf.TensorShape(shape), CLASS_NAMES.shape) 

    labeled_ds = Dataset.from_generator(gen, output_types, output_shapes)
    return labeled_ds


def prep_ds(dataset: Dataset, batch_size=16, cache:str=None, shuffle_buffer_size=1000, seed=None):
    if cache:
        if isinstance(cache, str):
            dataset = dataset.cache(cache)
        else:
            dataset = dataset.cache() #pylint: disable=no-member
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def prep_ds_train(dataset: Dataset, batch_size=16, cache=True, shuffle_buffer_size=1000, seed=None):
    # This is a small dataset, only load it once, and keep it in memory. Use `.cache(filename)` to cache preprocessing work for datasets that don't fit in memory.
    if cache:
        if isinstance(cache, str):
            dataset = dataset.cache(cache)
        else:
            dataset = dataset.cache() #pylint: disable=no-member
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE) # `prefetch` lets the dataset fetch batches in the background while the model is training.
    return dataset


def dataset_from_dir(
    dir_path: Union[str, Path],
    preprocess_input:Callable=None,
    shape=[224,224,3], 
    seed=1
) -> Dataset :
    dir_path = Path(dir_path)
    data_paths = [str(p) for suffix in ['jpg','mat'] for p in dir_path.glob('*/*'+suffix)]
    return dataset_from_paths(data_paths, preprocess_input, shape, seed), len(data_paths)


def balanced_dataset_split_from_dir(
    dir_path: Union[str, Path],
    samples_per_class: int,
    preprocess_input:Callable=None,
    shape=[224,224,3], 
    seed=1
) -> Tuple[Dataset, Dataset, int, int]:
    sampled_dataset, rest_dataset = [], []
    dir_path = Path(dir_path)
    class_paths = dir_path.glob('*')
    for class_path in class_paths:
        tmp = [str(p) for suffix in ['jpg','mat'] for p in class_path.glob('*'+suffix)]
        random.seed(seed)
        random.shuffle(tmp)
        sampled_dataset.extend(tmp[:samples_per_class])
        rest_dataset.extend(tmp[samples_per_class:])

    return ( 
        dataset_from_paths(sampled_dataset, preprocess_input, shape, seed), 
        dataset_from_paths(rest_dataset, preprocess_input, shape, seed), 
        len(sampled_dataset),
        len(rest_dataset),
    )


def balanced_dataset_tvt_split_from_dir(
    dir_path: Union[str, Path],
    samples_per_train_class: int,
    samples_per_val_class: int,
    preprocess_input:Callable=None,
    shape=[224,224,3], 
    seed=1
) -> Tuple[Dataset, Dataset, int, int]:
    train_dataset, val_dataset, test_dataset = [], [], []
    dir_path = Path(dir_path)
    class_paths = dir_path.glob('*')
    for class_path in class_paths:
        tmp = [str(p) for suffix in ['jpg','mat'] for p in class_path.glob('*'+suffix)]
        random.seed(seed)
        random.shuffle(tmp)
        train_dataset.extend(tmp[:samples_per_train_class])
        val_dataset.extend(tmp[samples_per_train_class:samples_per_train_class+samples_per_val_class])
        test_dataset.extend(tmp[samples_per_train_class+samples_per_val_class:])

    return ( 
        dataset_from_paths(train_dataset, preprocess_input, shape, seed), 
        dataset_from_paths(val_dataset, preprocess_input, shape, seed), 
        dataset_from_paths(test_dataset, preprocess_input, shape, seed), 
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )
             

def office31_datasets(
    source_name: str, 
    target_name: str, 
    preprocess_input:Callable=None,
    shape=[224,224,3], 
    seed=1,
    features='images'
) -> Dict[str, Dict[str, Dict[str, Tuple[Dataset, int]]]]:
    ''' Create the datasets needed for evaluating the Office31 dataset
    Returns:
        Dictionary of ['source'|'target'] ['full'|'train'|'test'] ['ds'|'size']
    '''
    name_mapping = {
        'A' : 'amazon',
        'D' : 'dslr',
        'W' : 'webcam',
    }

    if source_name in name_mapping.keys():
        source_name = name_mapping[source_name]
    elif source_name not in name_mapping.values():
        raise ValueError('source_name must be one of {}'.format(name_mapping.items()))
    if target_name in name_mapping.keys():
        target_name = name_mapping[target_name]
    elif target_name not in name_mapping.values():
        raise ValueError('source_name must be one of {}'.format(name_mapping.items()))

    project_base_path = Path(__file__).parent.parent
    source_data_path = project_base_path / 'datasets' / 'Office31' / source_name / features
    target_data_path = project_base_path / 'datasets' / 'Office31' / target_name / features

    dataset_configs = load_json(project_base_path / 'configs' / 'splits.json')
    n_source_samples = dataset_configs[source_name]['source_samples']
    n_target_samples = dataset_configs[target_name]['target_samples']
    n_target_val_samples = dataset_configs[target_name]['target_val_samples']

    s_full, s_full_size         = dataset_from_dir(source_data_path, preprocess_input, shape, seed)
    s_train, _, s_train_size, _ = balanced_dataset_split_from_dir(source_data_path, n_source_samples, preprocess_input, shape, seed)
    t_train, t_val, t_test, t_train_size, t_val_size, t_test_size \
        = balanced_dataset_tvt_split_from_dir(target_data_path, n_target_samples, n_target_val_samples, preprocess_input, shape, seed)

    return {
        'source': {
            'full': { 'ds': s_full, 'size': s_full_size },
            'train': { 'ds': s_train, 'size': s_train_size},
        },
        'target': {
            'train': { 'ds': t_train, 'size': t_train_size},
            'val': { 'ds': t_val, 'size': t_val_size},
            'test': { 'ds': t_test, 'size': t_test_size},
        },
    }


def office31_class_names() -> List[str]:
    data_dir = Path(__file__).parent.parent / 'datasets' / 'Office31' / 'amazon' / 'images'
    return sorted([item.name for item in data_dir.glob('*') if item.is_dir()])

def get_random_tf_seed():
    return tf.random.uniform(
        shape=tf.compat.v2.TensorShape([]), 
        maxval=tf.constant(value=9223372036854775807,dtype=tf.int64), 
        dtype=tf.int64
    ) 


def da_pair_dataset(
    source_ds,
    target_ds,
    ratio:Optional[float]=None,
    shuffle_buffer_size=5000,
    mdl_ins=['input_source', 'input_target'],
    mdl_outs=['preds', 'preds_1', 'aux_out']
) -> Tuple[Dataset, int]:
    ''' Create a paired dataset of positive and negative pairs from source and target datasets.
        NB: This has not been optimized for large datasets!
        NB: this assumes a certain naming of the model inputs and outputs (can be specified as parameter)
        @returns: the paired dataset and its size
    '''
    assert tf.executing_eagerly()

    source_ds = source_ds.shuffle(buffer_size=shuffle_buffer_size)
    target_ds = target_ds.shuffle(buffer_size=shuffle_buffer_size)

    def equal_tensors(t1, t2):
        return tf.cast(tf.math.reduce_all(tf.equal(t1, t2)), dtype=DTYPE)

    def count_pair_types(src_ds, tgt_ds):
        tot = 0
        pos = 0
        for (_, s_lbl), (_, t_lbl) in itertools.product(src_ds, tgt_ds):
            pos += int(equal_tensors(s_lbl, t_lbl).numpy())
            tot += 1
        neg = tot - pos
        return pos, neg

    n_pos, n_neg = count_pair_types(source_ds, target_ds)
    target_neg = round(n_pos*ratio)
    size_ds = n_pos + min(n_neg, target_neg)

    def gen_all():
        for (xs, ys), (xt, yt) in itertools.product(source_ds, target_ds):
            # yield xs, xt, ys, yt, [ys, yt]
            # yield (xs, xt), (ys, yt, [ys, yt])
            yield {mdl_ins[0]:xs, mdl_ins[1]:xt}, {mdl_outs[0]:ys, mdl_outs[1]:yt, mdl_outs[2]:[ys, yt]}

    def gen_ratio():
        if not ratio or target_neg > n_neg:
            return gen_all

        neg_left = target_neg
        for (xs, ys), (xt, yt) in itertools.product(source_ds, target_ds):
            eq = equal_tensors(ys, yt)
            if not eq.numpy():
                if neg_left > 0:
                    neg_left -= 1
                    # yield xs, xt, ys, yt, [ys, yt]
                    # yield (xs, xt), (ys, yt, [ys, yt])
                    yield {mdl_ins[0]:xs, mdl_ins[1]:xt}, {mdl_outs[0]:ys, mdl_outs[1]:yt, mdl_outs[2]:[ys, yt]}
            else:
                # yield xs, xt, ys, yt, [ys, yt]
                # yield (xs, xt), (ys, yt, [ys, yt])
                yield {mdl_ins[0]:xs, mdl_ins[1]:xt}, {mdl_outs[0]:ys, mdl_outs[1]:yt, mdl_outs[2]:[ys, yt]}

    shapes = ({ mdl_ins[0]:  source_ds.output_shapes[0], 
                mdl_ins[1]:  target_ds.output_shapes[0] }, 
              { mdl_outs[0]: source_ds.output_shapes[1], 
                mdl_outs[1]: target_ds.output_shapes[1], 
                mdl_outs[2]: tf.compat.v2.TensorShape([2,target_ds.output_shapes[1][0]])
              })

    types  = ({ mdl_ins[0]:  source_ds.output_types[0], 
                mdl_ins[1]:  target_ds.output_types[0] }, 
              { mdl_outs[0]: source_ds.output_types[1], 
                mdl_outs[1]: target_ds.output_types[1], 
                mdl_outs[2]: target_ds.output_types[1]
              })

    mix_ds = Dataset.from_generator(gen_ratio, types, shapes)#.shuffle(buffer_size=shuffle_buffer_size)
    return {
        'ds': mix_ds,
        'size':size_ds
    }


def da_pair_repeat_dataset(
    val_ds: Dict,
    mdl_ins=['input_source', 'input_target'],
    mdl_outs=['preds', 'preds_1', 'aux_out']
) -> Tuple[Dataset, int]:
    ''' Create a paired dataset by repeating the data on two streams
        NB: this assumes a certain naming of the model inputs and outputs (can be specified as parameter)
        @returns: the paired dataset and its size
    '''
    assert tf.executing_eagerly()

    ds = val_ds['ds']
    size_ds = val_ds['size']

    def gen_pars():
        for (d, l) in ds:
            yield {mdl_ins[0]:d, mdl_ins[1]:d}, {mdl_outs[0]:l, mdl_outs[1]:l, mdl_outs[2]:[l,l]}

    shapes = ({ mdl_ins[0]:  ds.output_shapes[0], 
                mdl_ins[1]:  ds.output_shapes[0] }, 
              { mdl_outs[0]: ds.output_shapes[1], 
                mdl_outs[1]: ds.output_shapes[1], 
                mdl_outs[2]: tf.compat.v2.TensorShape([2,ds.output_shapes[1][0]]) 
              })

    types  = ({ mdl_ins[0]:  ds.output_types[0], 
                mdl_ins[1]:  ds.output_types[0] }, 
              { mdl_outs[0]: ds.output_types[1], 
                mdl_outs[1]: ds.output_types[1], 
                mdl_outs[2]: tf.bool })

    pair_ds = Dataset.from_generator(gen_pars, types, shapes)
    return {
        'ds': pair_ds,
        'size':size_ds
    }
    

def make_ds_example(xs, xt, ys, yt): 
    return {
        'input_source':xs, 
        'input_target':xt, 
        'label_source':ys, 
        'label_target':yt
    }

def make_ds_shapes(source_ds, target_ds):
    return {  
        'input_source': source_ds.output_shapes[0], 
        'input_target': target_ds.output_shapes[0], 
        'label_source': source_ds.output_shapes[1], 
        'label_target': target_ds.output_shapes[1], 
    }

def make_ds_types(source_ds, target_ds):
    return {  
        'input_source': source_ds.output_types[0], 
        'input_target': target_ds.output_types[0] , 
        'label_source': source_ds.output_types[1], 
        'label_target': target_ds.output_types[1], 
    }

def da_pair_alt_dataset(
    source_ds,
    target_ds,
    ratio:Optional[float]=None,
    shuffle_buffer_size=5000,
) -> Tuple[Dataset, int]:
    ''' Create a paired dataset of positive and negative pairs from source and target datasets.
        NB: This has not been optimized for large datasets!
        NB: this assumes a certain naming of the model inputs and outputs (can be specified as parameter)
        @returns: the paired dataset and its size
    '''
    assert tf.executing_eagerly()

    source_ds = source_ds.shuffle(buffer_size=shuffle_buffer_size)
    target_ds = target_ds.shuffle(buffer_size=shuffle_buffer_size)

    def equal_tensors(t1, t2):
        return tf.cast(tf.math.reduce_all(tf.equal(t1, t2)), dtype=DTYPE)

    def count_pair_types(src_ds, tgt_ds):
        tot = 0
        pos = 0
        for (_, s_lbl), (_, t_lbl) in itertools.product(src_ds, tgt_ds):
            pos += int(equal_tensors(s_lbl, t_lbl).numpy())
            tot += 1
        neg = tot - pos
        return pos, neg

    n_pos, n_neg = count_pair_types(source_ds, target_ds)
    target_neg = round(n_pos*ratio)
    size_ds = n_pos + min(n_neg, target_neg)

    def gen_all():
        for (xs, ys), (xt, yt) in itertools.product(source_ds, target_ds):
            yield make_ds_example(xs, xt, ys, yt)

    def gen_ratio():
        if not ratio or target_neg > n_neg:
            return gen_all

        neg_left = target_neg
        for (xs, ys), (xt, yt) in itertools.product(source_ds, target_ds):
            eq = equal_tensors(ys, yt)
            if not eq.numpy():
                if neg_left > 0:
                    neg_left -= 1
                    yield make_ds_example(xs, xt, ys, yt)
            else:
                yield make_ds_example(xs, xt, ys, yt)

    shapes = make_ds_shapes(source_ds, target_ds)
    types  = make_ds_types(source_ds, target_ds)

    mix_ds = Dataset.from_generator(gen_ratio, types, shapes)#.shuffle(buffer_size=shuffle_buffer_size)
    return {
        'ds': mix_ds,
        'size':size_ds
    }


def da_pair_alt_repeat_dataset(
    val_ds: Dict,
    mdl_ins=['input_source', 'input_target'],
    mdl_outs=['preds', 'preds_1', 'aux_out']
) -> Tuple[Dataset, int]:
    ''' Create a paired dataset by repeating the data on two streams
        NB: this assumes a certain naming of the model inputs and outputs (can be specified as parameter)
        @returns: the paired dataset and its size
    '''
    assert tf.executing_eagerly()

    ds = val_ds['ds']
    size_ds = val_ds['size']

    def gen_pars():
        for (d, l) in ds:
            yield make_ds_example(d, d, l, l) 

    shapes = make_ds_shapes(ds, ds)
    types  = make_ds_types(ds, ds)

    pair_ds = Dataset.from_generator(gen_pars, types, shapes)
    return {
        'ds'  : pair_ds,
        'size': size_ds
    }



def flip(x: tf.Tensor) -> tf.Tensor:
        return tf.image.random_flip_left_right(x)

def color(x: tf.Tensor) -> tf.Tensor:
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x

def rotate(x: tf.Tensor) -> tf.Tensor:
    max_rot = PI / 45
    # return tfa.image.transform_ops.rotate(x, tf.random.uniform(shape=[], minval=-max_rot, maxval=max_rot, dtype=DTYPE), interpolation='BILINEAR')
    return tf.contrib.image.rotate(x, tf.random.uniform(shape=[], minval=-max_rot, maxval=max_rot, dtype=DTYPE), interpolation='BILINEAR')

def zoom(x: tf.Tensor, batch_size=16, crop_size=(224,224)) -> tf.Tensor:
    scales = np.linspace(0.8, 1.0, batch_size)
    np.random.shuffle(scales)
    boxes = np.zeros((len(scales), 4))
    for i, scale in enumerate(list(scales)):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]
    box_ind = np.arange(0, batch_size)
    return tf.image.crop_and_resize(x, boxes=boxes, box_indices=box_ind, crop_size=crop_size)

def clip(x: tf.Tensor) -> tf.Tensor:
    return tf.clip_by_value(x, 0, 1)

def augment(dataset:Dataset, batch_size=16, crop_size=(224,224)):
    for f in [
        flip, 
        color, 
        rotate, 
        partial(zoom, batch_size=batch_size, crop_size=crop_size),
        # clip
    ]:
        dataset = dataset.map(
            map_func=lambda x, y: tf.cond(
                pred=tf.random.uniform([], 0, 1) > 0.5, 
                true_fn=lambda: (f(x), y),
                false_fn=lambda: (x,y)
            ), 
            num_parallel_calls=AUTOTUNE
        )
    return dataset

def augment_pair(
    dataset:Dataset, 
    batch_size=16, 
    crop_size=(224,224),
    mdl_ins=['input_source', 'input_target'],
    mdl_outs=['preds', 'preds_1', 'aux_out']
):
    for f in [
        flip, 
        color, 
        rotate, 
        partial(zoom, batch_size=batch_size, crop_size=crop_size),
        # clip
    ]:
        dataset = dataset.map(
            map_func=lambda x, y: tf.cond(
                pred=tf.random.uniform([], 0, 1) > 0.5, 
                true_fn=lambda: ( {k:f(x[k]) for k in mdl_ins}, y ),
                false_fn=lambda: (x, y),
            ), 
            num_parallel_calls=AUTOTUNE
        )
    return dataset
