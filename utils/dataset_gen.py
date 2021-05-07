from __future__ import absolute_import, division, print_function, unicode_literals

import itertools
import random
from functools import partial, reduce

# import tensorflow_addons as tfa
from math import pi as PI
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import datasetops as do
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from scipy.io import loadmat
from torchvision.datasets import USPS
from utils.mnist_m import MnistM  # noqa: F401

from utils.file_io import load_json

AUTOTUNE = tf.data.experimental.AUTOTUNE
Dataset = tf.compat.v2.data.Dataset
DTYPE = tf.float32
ImageShape = Tuple[int, int, int]

OFFICE_DICT = {
    "A": "amazon",
    "D": "dslr",
    "W": "webcam",
}

DIGITS_DICT = {"M": "mnist", "U": "usps", "Mm": "mnist-m", "S": "svhn"}

OFFICE_DATASET_NAMES = ["A", "D", "W", "amazon", "dslr", "webcam"]
DIGIT_DATASET_NAMES = ["M", "U", "Mm", "S", "mnist", "usps", "mnist_m", "svhn"]


DIGITS_MEAN = {
    "mnist": (33.31842145),
    "usps": (63.33399645),
    "mnist_m": (116.76592523, 117.83341324, 104.10233177),
    "svhn": (115.3679051, 115.38643995, 119.58916846),
}

DIGITS_STD = {
    "mnist": (78.56748998),
    "usps": (93.39424878),
    "mnist_m": (64.24722818, 60.37840705, 65.96992975),
    "svhn": (55.95578582, 57.77526543, 58.26906232),
}

DIGITS_SHAPE: Dict[str, ImageShape] = {
    "mnist": (28, 28, 1),
    "usps": (16, 16, 1),
    "mnist_m": (32, 32, 3),
    "svhn": (32, 32, 3),
}


def digits_shape(source: str, target: str, mode: int = 1) -> ImageShape:
    if source in DIGITS_DICT.keys():
        source = DIGITS_DICT[source]
    if target in DIGITS_DICT.keys():
        target = DIGITS_DICT[target]

    size = {
        1: DIGITS_SHAPE[source][0],
        2: DIGITS_SHAPE[target][0],
        3: min(DIGITS_SHAPE[source][0], DIGITS_SHAPE[target][0]),
        4: max(DIGITS_SHAPE[source][0], DIGITS_SHAPE[target][0]),
    }[mode]

    num_channels = max(DIGITS_SHAPE[source][2], DIGITS_SHAPE[target][2])

    return (size, size, num_channels)


def make_image_prep(
    shape=[224, 224, 3],
    preprocess_input_fn: Callable = None,
):
    shape = shape[:2]

    def prep(file_path):
        if isinstance(file_path, str):
            file_path = tf.constant(file_path)
        img = tf.compat.v2.io.read_file(
            file_path
        )  # load the raw data from the file as a string
        img = tf.compat.v2.image.decode_jpeg(
            img, channels=3
        )  # convert the compressed string to a 3D uint8 tensor
        img = tf.compat.v2.image.convert_image_dtype(
            img, DTYPE
        )  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        if preprocess_input_fn:
            img = preprocess_input_fn(
                img * 255
            )  # preprocess input assumes input in [0,255] range.
        img = tf.compat.v2.image.resize(
            img, shape
        )  # resize the image to the desired size.
        return img

    return prep


def make_mat_prep(
    shape=[224, 224, 3],
    preprocess_input_fn: Callable = None,
):
    def prep(file_path):
        d = loadmat(file_path)
        if preprocess_input_fn:
            d = preprocess_input_fn(
                d
            )  # should pick the right column from the dict above
        return d

    return prep


def dataset_from_paths(
    data_paths: List[str],
    preprocess_input: Callable = None,
    shape=[224, 224, 3],
):
    CLASS_NAMES = tf.constant(sorted(list(set([p.split("/")[-2] for p in data_paths]))))
    data_type = Path(data_paths[0]).suffix if data_paths else "none"

    prep_dat = {
        # '.jpg': lambda p, fn=make_image_prep(shape, preprocess_input): fn(tf.constant(p, dtype=tf.string)),
        ".jpg": make_image_prep(shape, preprocess_input),
        ".mat": make_mat_prep(shape, preprocess_input),
        "none": lambda x: x,
    }[data_type]

    def prep_label(file_path):
        fp = tf.constant(file_path)
        parts = tf.compat.v2.strings.split(
            fp, "/"
        )  # convert the path to a list of path components
        label = tf.equal(
            CLASS_NAMES, parts[-2]
        )  # The second to last is the class-directory
        return label

    def gen():
        for fp in data_paths:
            yield prep_dat(fp), prep_label(fp)

    output_types = (DTYPE, tf.bool)
    output_shapes = (tf.TensorShape(shape), CLASS_NAMES.shape)

    labeled_ds = Dataset.from_generator(gen, output_types, output_shapes)
    return labeled_ds


def prep_ds(
    dataset: Dataset,
    batch_size=16,
    cache: str = None,
    shuffle_buffer_size=1000,
    seed=None,
):
    if cache:
        if isinstance(cache, str):
            dataset = dataset.cache(cache)
        else:
            dataset = dataset.cache()  # pylint: disable=no-member
    # dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def prep_ds_train(
    dataset: Dataset, batch_size=16, cache=True, shuffle_buffer_size=1000, seed=None
):
    # This is a small dataset, only load it once, and keep it in memory. Use `.cache(filename)` to cache preprocessing work for datasets that don't fit in memory.
    if cache:
        if isinstance(cache, str):
            dataset = dataset.cache(cache)
        else:
            dataset = dataset.cache()  # pylint: disable=no-member
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(
        buffer_size=AUTOTUNE
    )  # `prefetch` lets the dataset fetch batches in the background while the model is training.
    return dataset


def dataset_from_dir(
    dir_path: Union[str, Path],
    preprocess_input: Callable = None,
    shape=[224, 224, 3],
    seed=1,
) -> Dataset:
    dir_path = Path(dir_path)
    data_paths = [
        str(p) for suffix in ["jpg", "mat"] for p in dir_path.glob("*/*" + suffix)
    ]
    return dataset_from_paths(data_paths, preprocess_input, shape), len(data_paths)


def balanced_dataset_split_from_dir(
    dir_path: Union[str, Path],
    samples_per_class: int,
    preprocess_input: Callable = None,
    shape=[224, 224, 3],
    seed=1,
) -> Tuple[Dataset, Dataset, int, int]:
    sampled_dataset, rest_dataset = [], []
    dir_path = Path(dir_path)
    class_paths = dir_path.glob("*")
    for class_path in class_paths:
        tmp = [
            str(p) for suffix in ["jpg", "mat"] for p in class_path.glob("*" + suffix)
        ]
        random.seed(seed)
        random.shuffle(tmp)
        sampled_dataset.extend(tmp[:samples_per_class])
        rest_dataset.extend(tmp[samples_per_class:])

    return (
        dataset_from_paths(sampled_dataset, preprocess_input, shape),
        dataset_from_paths(rest_dataset, preprocess_input, shape),
        len(sampled_dataset),
        len(rest_dataset),
    )


def balanced_dataset_tvt_split_from_dir(
    dir_path: Union[str, Path],
    samples_per_train_class: int,
    samples_per_val_class: int,
    preprocess_input: Callable = None,
    shape=[224, 224, 3],
    seed=1,
) -> Tuple[Dataset, Dataset, Dataset, int, int, int]:
    train_dataset, val_dataset, test_dataset = [], [], []
    dir_path = Path(dir_path)
    class_paths = dir_path.glob("*")
    for class_path in class_paths:
        tmp = [
            str(p) for suffix in ["jpg", "mat"] for p in class_path.glob("*" + suffix)
        ]
        random.seed(seed)
        random.shuffle(tmp)
        train_dataset.extend(tmp[:samples_per_train_class])
        val_dataset.extend(
            tmp[
                samples_per_train_class : samples_per_train_class
                + samples_per_val_class
            ]
        )
        test_dataset.extend(tmp[samples_per_train_class + samples_per_val_class :])

    return (
        dataset_from_paths(train_dataset, preprocess_input, shape),
        dataset_from_paths(val_dataset, preprocess_input, shape),
        dataset_from_paths(test_dataset, preprocess_input, shape),
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )


def split_classwise(ds: Dataset, labels: List[int]) -> List[Dataset]:
    return [
        ds.filter(lambda i, l: tf.equal(tf.argmax(l, axis=0), lbl))  # type: ignore
        for lbl in labels
    ]


def split(
    ds_classwise: Iterable[Dataset], take_num: int
) -> List[Tuple[Dataset, Dataset]]:
    return [
        # (selected, rest)
        (d.take(take_num), d.skip(take_num))
        for d in ds_classwise
    ]


def multi_split(ds_classwise: Iterable[Dataset], take_num: List[int]):
    if len(take_num) == 0:
        return ds_classwise
    if len(take_num) == 1:
        return split(ds_classwise, take_num[0])
    if len(take_num) > 1:
        split_wise = list(zip(*split(ds_classwise, take_num[0])))
        return [
            (a, *bc)
            for (a, bc) in list(
                zip(  # type:ignore
                    split_wise[0], multi_split(split_wise[1], take_num[1:])
                )
            )
        ]


def collect_classwise_splits(classwise_split_ds):
    return [
        reduce(lambda x, acc: acc.concatenate(x), slds)
        for slds in list(zip(*classwise_split_ds))
    ]


def balanced_splits(ds, split_num: List[int], labels: List[Any]):
    return collect_classwise_splits(multi_split(split_classwise(ds, labels), split_num))


def one_hot(x: tf.uint64, depth=10):
    y = tf.one_hot(x, depth=depth)
    # y = tf.cast(y, tf.bool)
    return y


def preprocess_digits(dataset_name: str, shape: ImageShape, standardize=True):
    if dataset_name in DIGITS_DICT.keys():
        dataset_name = DIGITS_DICT[dataset_name]

    mean, std = DIGITS_MEAN[dataset_name], DIGITS_STD[dataset_name]

    def fn(x):
        im, lbl = x["image"], x["label"]
        im = tf.cast(im, tf.float32)
        if standardize:
            im = (im - mean) / std
        else:
            im = im / 255
        im = tf.image.resize(im, shape[:-1])
        im = tf.broadcast_to(im, shape)
        lbl = one_hot(lbl)
        return (im, lbl)

    return fn


def digits_datasets(
    source_name: str,
    target_name: str,
    num_source_samples_per_class: int,
    num_target_samples_per_class: int,
    num_val_samples_per_class: int,
    input_shape: ImageShape,
    standardize_input=True,
    shuffle_buffer_size=1000,
    seed=1,
    test_as_val=False,
) -> Dict[str, Dict[str, Tuple[Dataset, int]]]:

    class_names = digits_class_names()
    num_classes = len(class_names)

    t_data, t_info = tfds.load(target_name, split="train", with_info=True)
    s_data, s_info = tfds.load(source_name, split="train", with_info=True)

    s_data = s_data.map(
        preprocess_digits(source_name, input_shape, standardize_input), AUTOTUNE
    )
    t_data = t_data.map(
        preprocess_digits(target_name, input_shape, standardize_input), AUTOTUNE
    )

    s_full_size = s_info.splits["train"].num_examples
    s_data.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
    s_train, _ = balanced_splits(s_data, [num_source_samples_per_class], class_names)
    s_train_size = num_source_samples_per_class * num_classes

    if not test_as_val:
        t_train, t_val, t_test = balanced_splits(
            t_data,
            [num_target_samples_per_class, num_val_samples_per_class],
            class_names,
        )
        t_val_size = num_val_samples_per_class * num_classes
    else:
        t_train, t_test = balanced_splits(
            t_data, [num_target_samples_per_class], class_names
        )
        t_val, t_val_size = None, 0

    t_train_size = num_target_samples_per_class * num_classes
    t_test_size = t_info.splits["train"].num_examples - t_train_size - t_val_size

    return {
        "source": {
            "full": (s_data, s_full_size),
            "train": (s_train, s_train_size),
            "shape": s_info.features["image"].shape,
        },
        "target": {
            "train": (t_train, t_train_size),
            "val": (t_val, t_val_size),
            "test": (t_test, t_test_size),
            "shape": t_info.features["image"].shape,
        },
    }


def digits_datasets_new(
    source_name: str,
    target_name: str,
    num_source_samples_per_class: int,
    num_target_samples_per_class: int,
    num_val_samples_per_class: int,
    input_shape: ImageShape,
    standardize_input=True,
    shuffle_buffer_size=1000,
    seed=1,
) -> Dict[str, Dict[str, Tuple[Dataset, int]]]:

    class_names = digits_class_names()
    num_classes = len(class_names)

    def get_dataset(dataset_name: str):
        dataset_name = dataset_name.lower()
        if dataset_name in ["mnist", "svhn", "mnist_m"]:
            if dataset_name == "svhn":
                dataset_name = "svhn_cropped"  # tfds name
            train, info = tfds.load(dataset_name, split="train", with_info=True)
            test = tfds.load(dataset_name, split="test", with_info=False)
            train_size = info.splits["train"].num_examples
            test_size = info.splits["test"].num_examples
        elif dataset_name == "usps":
            datasets_path = str((Path(__file__).parent.parent / "datasets").absolute())
            ds_train = do.from_pytorch(USPS(datasets_path, download=True, train=True))
            ds_test = do.from_pytorch(
                USPS(
                    datasets_path,
                    download=True,
                    train=False,
                )
            )
            train_size = len(ds_train)
            test_size = len(ds_test)
            train = ds_train.transform(
                lambda x: {
                    "image": np.expand_dims(np.array(x[0]), axis=2),
                    "label": x[1],
                }
            ).to_tensorflow()
            test = ds_train.transform(
                lambda x: {
                    "image": np.expand_dims(np.array(x[0]), axis=2),
                    "label": x[1],
                }
            ).to_tensorflow()
        else:
            raise ValueError(
                "dataset should be one of ['mnist','mnist_m','svhn','usps']"
            )
        return train, test, train_size, test_size

    s_train, s_test, s_train_size, s_test_size = get_dataset(source_name)
    t_train, t_test, t_train_size, t_test_size = get_dataset(target_name)

    # preprocess
    s_train = s_train.map(
        preprocess_digits(source_name, input_shape, standardize_input), AUTOTUNE
    )
    t_train = t_train.map(
        preprocess_digits(target_name, input_shape, standardize_input), AUTOTUNE
    )
    t_test = t_test.map(
        preprocess_digits(target_name, input_shape, standardize_input), AUTOTUNE
    )

    # source split
    s_train.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
    if num_source_samples_per_class > 0:
        s_train_sampled, _ = balanced_splits(
            s_train, [num_source_samples_per_class], class_names
        )
        s_train_sampled_size = num_source_samples_per_class * num_classes
    else:
        s_train_sampled = s_train
        s_train_sampled_size = s_train_size

    # target split
    t_train_sampled, t_val = balanced_splits(
        t_train,
        [num_target_samples_per_class],
        class_names,
    )
    t_train_sampled_size = num_target_samples_per_class * num_classes
    t_val_size = t_train_size - t_train_sampled_size

    return {
        "source": {
            "full": (s_train, s_train_size),
            "train": (s_train_sampled, s_train_sampled_size),
        },
        "target": {
            "train": (t_train_sampled, t_train_sampled_size),
            "val": (t_val, t_val_size),
            "test": (t_test, t_test_size),
        },
    }


def office31_datasets_new(
    source_name: str,
    target_name: str,
    shape=[224, 224, 3],
    preprocess_input: Callable = None,
    seed=1,
) -> Dict[str, Dict[str, Tuple[Dataset, int]]]:
    """Create the datasets needed for evaluating the Office31 dataset
    Returns:
        Dictionary of ['source'|'target'] ['full'|'train'|'test'] ['ds'|'size']
    """
    if source_name in OFFICE_DICT.keys():
        source_name = OFFICE_DICT[source_name]
    elif source_name not in OFFICE_DICT.values():
        raise ValueError("source_name must be one of {}".format(OFFICE_DICT.items()))
    if target_name in OFFICE_DICT.keys():
        target_name = OFFICE_DICT[target_name]
    elif target_name not in OFFICE_DICT.values():
        raise ValueError("source_name must be one of {}".format(OFFICE_DICT.items()))

    project_base_path = Path(__file__).parent.parent
    source_data_path = (
        project_base_path / "datasets" / "Office31" / source_name / "images"
    )
    target_data_path = (
        project_base_path / "datasets" / "Office31" / target_name / "images"
    )

    source = do.from_folder_class_data(source_data_path).named("s_data", "s_label")
    target = do.from_folder_class_data(target_data_path).named("t_data", "t_label")

    num_source_per_class = 20 if "amazon" in str(source_data_path) else 8
    num_target_per_class = 3

    source_train = source.shuffle(seed).filter(
        s_label=do.allow_unique(num_source_per_class)
    )

    target_test, target_trainval = target.shuffle(42).split(
        fractions=[0.3, 0.7], seed=42  # hard-coded seed
    )
    target_train, target_val = target_trainval.shuffle(seed).split_filter(
        t_label=do.allow_unique(num_target_per_class)
    )

    # ensure that all one_hot mapping are the same
    def make_one_hot_mapping():
        d = {k: i for i, k in enumerate(sorted(target_train.unique(1)))}

        def fn(key):
            return d[key]

        return fn

    one_hot_mapping_fn = make_one_hot_mapping()

    # transform all data to use a one-hot encoding for the label
    source, source_train, target_train, target_val, target_test = [
        d.named("data", "label").transform(
            data=[do.image_resize(shape[:2]), do.numpy(), preprocess_input],
            label=do.one_hot(encoding_size=31, mapping_fn=one_hot_mapping_fn),
        )
        for d in [source, source_train, target_train, target_val, target_test]
    ]

    return {
        "source": {
            "full": (source.to_tensorflow(), len(source)),
            "train": (source_train.to_tensorflow(), len(source_train)),
        },
        "target": {
            "train": (target_train.to_tensorflow(), len(target_train)),
            "val": (target_val.to_tensorflow(), len(target_val)),
            "test": (target_test.to_tensorflow(), len(target_test)),
        },
    }


def office31_datasets(
    source_name: str,
    target_name: str,
    preprocess_input: Callable = None,
    shape=[224, 224, 3],
    seed=1,
    features="images",
    test_as_val=False,
) -> Dict[str, Dict[str, Tuple[Dataset, int]]]:
    """Create the datasets needed for evaluating the Office31 dataset
    Returns:
        Dictionary of ['source'|'target'] ['full'|'train'|'test'] ['ds'|'size']
    """

    if source_name in OFFICE_DICT.keys():
        source_name = OFFICE_DICT[source_name]
    elif source_name not in OFFICE_DICT.values():
        raise ValueError("source_name must be one of {}".format(OFFICE_DICT.items()))
    if target_name in OFFICE_DICT.keys():
        target_name = OFFICE_DICT[target_name]
    elif target_name not in OFFICE_DICT.values():
        raise ValueError("source_name must be one of {}".format(OFFICE_DICT.items()))

    project_base_path = Path(__file__).parent.parent
    source_data_path = (
        project_base_path / "datasets" / "Office31" / source_name / features
    )
    target_data_path = (
        project_base_path / "datasets" / "Office31" / target_name / features
    )

    dataset_configs = load_json(project_base_path / "configs" / "splits.json")
    n_source_samples = dataset_configs[source_name]["source_samples"]
    n_target_samples = dataset_configs[target_name]["target_samples"]
    n_target_val_samples = dataset_configs[target_name]["target_val_samples"]

    s_full, s_full_size = dataset_from_dir(
        source_data_path, preprocess_input, shape, seed
    )
    s_train, _, s_train_size, _ = balanced_dataset_split_from_dir(
        source_data_path, n_source_samples, preprocess_input, shape, seed
    )

    if not test_as_val:
        (
            t_train,
            t_val,
            t_test,
            t_train_size,
            t_val_size,
            t_test_size,
        ) = balanced_dataset_tvt_split_from_dir(
            target_data_path,
            n_target_samples,
            n_target_val_samples,
            preprocess_input,
            shape,
            seed,
        )
    else:
        t_train, t_test, t_train_size, t_test_size = balanced_dataset_split_from_dir(
            target_data_path, n_target_samples, preprocess_input, shape, seed
        )
        t_val, t_val_size = None, 0

    return {
        "source": {
            "full": (s_full, s_full_size),
            "train": (s_train, s_train_size),
        },
        "target": {
            "train": (t_train, t_train_size),
            "val": (t_val, t_val_size),
            "test": (t_test, t_test_size),
        },
    }


def office31_class_names() -> List[str]:
    data_dir = (
        Path(__file__).parent.parent / "datasets" / "Office31" / "amazon" / "images"
    )
    return sorted([item.name for item in data_dir.glob("*") if item.is_dir()])


def digits_class_names() -> List[int]:
    return list(range(10))


def get_random_tf_seed():
    return tf.random.uniform(
        shape=tf.compat.v2.TensorShape([]),
        maxval=tf.constant(value=9223372036854775807, dtype=tf.int64),
        dtype=tf.int64,
    )


def da_pair_dataset(
    source_ds,
    target_ds,
    num_source_samples_per_class: int = None,
    num_target_samples_per_class: int = None,
    num_classes: int = None,
    ratio: Optional[float] = None,
    shuffle_buffer_size=5000,
    mdl_ins=["input_source", "input_target"],
    mdl_outs=["preds", "preds_1", "aux_out"],
) -> Tuple[Dataset, int]:
    """Create a paired dataset of positive and negative pairs from source and target datasets.
    NB: This has not been optimized for large datasets!
    NB: this assumes a certain naming of the model inputs and outputs (can be specified as parameter)
    @returns: the paired dataset and its size
    """
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
        print(f"pos pairs: {pos}")
        print(f"neg pairs: {neg}")
        return pos, neg

    # assumes that data is balanced (equal number of data per class)
    if not (
        num_source_samples_per_class > 0
        and num_source_samples_per_class
        and num_classes
    ):
        n_pos, n_neg = count_pair_types(source_ds, target_ds)
    else:
        n_pos = (
            num_classes
            * 1
            * num_source_samples_per_class
            * num_target_samples_per_class
        )
        n_neg = (
            num_classes
            * (num_classes - 1)
            * num_source_samples_per_class
            * num_source_samples_per_class
        )
    target_neg = round(n_pos * ratio) if ratio else n_neg
    size_ds = n_pos + target_neg

    def gen_all():
        for (xs, ys), (xt, yt) in itertools.product(source_ds, target_ds):
            # yield xs, xt, ys, yt, [ys, yt]
            # yield (xs, xt), (ys, yt, [ys, yt])
            yield {mdl_ins[0]: xs, mdl_ins[1]: xt}, {
                mdl_outs[0]: ys,
                mdl_outs[1]: yt,
                mdl_outs[2]: [ys, yt],
            }

    def gen_ratio():
        if not ratio or target_neg > n_neg:
            return gen_all

        neg_left = target_neg
        for (xs, ys), (xt, yt) in sorted(
            itertools.product(source_ds, target_ds), key=lambda k: random.random()
        ):
            eq = equal_tensors(ys, yt)
            if not eq.numpy():
                if neg_left > 0:
                    neg_left -= 1
                    yield {mdl_ins[0]: xs, mdl_ins[1]: xt}, {
                        mdl_outs[0]: ys,
                        mdl_outs[1]: yt,
                        mdl_outs[2]: [ys, yt],
                    }
            else:
                yield {mdl_ins[0]: xs, mdl_ins[1]: xt}, {
                    mdl_outs[0]: ys,
                    mdl_outs[1]: yt,
                    mdl_outs[2]: [ys, yt],
                }

    source_output_shapes = (
        list(source_ds.output_shapes.values())
        if hasattr(source_ds.output_shapes, "values")
        else source_ds.output_shapes
    )
    target_output_shapes = (
        list(target_ds.output_shapes.values())
        if hasattr(target_ds.output_shapes, "values")
        else target_ds.output_shapes
    )
    source_output_types = (
        list(source_ds.output_types.values())
        if hasattr(source_ds.output_types, "values")
        else source_ds.output_types
    )
    target_output_types = (
        list(target_ds.output_types.values())
        if hasattr(target_ds.output_types, "values")
        else target_ds.output_types
    )

    shapes = (
        {mdl_ins[0]: source_output_shapes[0], mdl_ins[1]: target_output_shapes[0]},
        {
            mdl_outs[0]: source_output_shapes[1],
            mdl_outs[1]: target_output_shapes[1],
            mdl_outs[2]: tf.compat.v2.TensorShape(
                [2, *target_output_shapes[1].as_list()]
            ),
        },
    )

    types = (
        {mdl_ins[0]: source_output_types[0], mdl_ins[1]: target_output_types[0]},
        {
            mdl_outs[0]: source_output_types[1],
            mdl_outs[1]: target_output_types[1],
            mdl_outs[2]: target_output_types[1],
        },
    )

    mix_ds = Dataset.from_generator(
        gen_ratio, types, shapes
    )  # .shuffle(buffer_size=shuffle_buffer_size)
    return (mix_ds, size_ds)


def da_pair_repeat_dataset(
    ds: Dataset,
    ds_size: int,
    mdl_ins=["input_source", "input_target"],
    mdl_outs=["preds", "preds_1", "aux_out"],
) -> Tuple[Dataset, int]:
    """Create a paired dataset by repeating the data on two streams
    NB: this assumes a certain naming of the model inputs and outputs (can be specified as parameter)
    @returns: the paired dataset and its size
    """
    assert tf.executing_eagerly()

    def gen_pars():
        for (d, l) in ds:
            yield {mdl_ins[0]: d, mdl_ins[1]: d}, {
                mdl_outs[0]: l,
                mdl_outs[1]: l,
                mdl_outs[2]: [l, l],
            }

    output_shapes = (
        list(ds.output_shapes.values())
        if hasattr(ds.output_shapes, "values")
        else ds.output_shapes
    )
    output_types = (
        list(ds.output_types.values())
        if hasattr(ds.output_types, "values")
        else ds.output_types
    )

    shapes = (
        {mdl_ins[0]: output_shapes[0], mdl_ins[1]: output_shapes[0]},
        {
            mdl_outs[0]: output_shapes[1],
            mdl_outs[1]: output_shapes[1],
            mdl_outs[2]: tf.compat.v2.TensorShape([2, *output_shapes[1].as_list()]),
        },
    )

    types = (
        {mdl_ins[0]: output_types[0], mdl_ins[1]: output_types[0]},
        {
            mdl_outs[0]: output_types[1],
            mdl_outs[1]: output_types[1],
            mdl_outs[2]: tf.bool,
        },
    )

    pair_ds = Dataset.from_generator(gen_pars, types, shapes)
    return (pair_ds, ds_size)


def make_ds_example(xs, xt, ys, yt):
    return {
        "input_source": xs,
        "input_target": xt,
        "label_source": ys,
        "label_target": yt,
    }


def make_ds_shapes(source_ds, target_ds):
    return {
        "input_source": source_ds.output_shapes[0],
        "input_target": target_ds.output_shapes[0],
        "label_source": source_ds.output_shapes[1],
        "label_target": target_ds.output_shapes[1],
    }


def make_ds_types(source_ds, target_ds):
    return {
        "input_source": source_ds.output_types[0],
        "input_target": target_ds.output_types[0],
        "label_source": source_ds.output_types[1],
        "label_target": target_ds.output_types[1],
    }


def da_pair_alt_dataset(
    source_ds,
    target_ds,
    ratio: Optional[float] = None,
    shuffle_buffer_size=5000,
) -> Tuple[Dataset, int]:
    """Create a paired dataset of positive and negative pairs from source and target datasets.
    NB: This has not been optimized for large datasets!
    NB: this assumes a certain naming of the model inputs and outputs (can be specified as parameter)
    @returns: the paired dataset and its size
    """
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
    target_neg = round(n_pos * ratio)
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
    types = make_ds_types(source_ds, target_ds)

    mix_ds = Dataset.from_generator(
        gen_ratio, types, shapes
    )  # .shuffle(buffer_size=shuffle_buffer_size)
    return (mix_ds, size_ds)


def da_pair_alt_repeat_dataset(
    ds: Dataset,
    ds_size: int,
    mdl_ins=["input_source", "input_target"],
    mdl_outs=["preds", "preds_1", "aux_out"],
) -> Tuple[Dataset, int]:
    """Create a paired dataset by repeating the data on two streams
    NB: this assumes a certain naming of the model inputs and outputs (can be specified as parameter)
    @returns: the paired dataset and its size
    """
    assert tf.executing_eagerly()

    def gen_pars():
        for (d, l) in ds:
            yield make_ds_example(d, d, l, l)

    shapes = make_ds_shapes(ds, ds)
    types = make_ds_types(ds, ds)

    pair_ds = Dataset.from_generator(gen_pars, types, shapes)
    return (pair_ds, ds_size)


def flip(x: tf.Tensor) -> tf.Tensor:
    return tf.image.random_flip_left_right(x)


def color(num_chan: int):
    def fn(x: tf.Tensor) -> tf.Tensor:
        if num_chan == 3:
            x = tf.image.random_hue(x, 0.08)
            x = tf.image.random_saturation(x, 0.6, 1.6)
        x = tf.image.random_brightness(x, 0.05)
        x = tf.image.random_contrast(x, 0.7, 1.3)
        return x

    return fn


def rotate(x: tf.Tensor) -> tf.Tensor:
    max_rot = PI / 45

    return tf.contrib.image.rotate(
        x,
        tf.random.uniform(shape=[], minval=-max_rot, maxval=max_rot, dtype=DTYPE),
        interpolation="BILINEAR",
    )


def zoom(x: tf.Tensor, batch_size=16, crop_size=(224, 224)) -> tf.Tensor:
    scales = np.linspace(0.8, 1.0, batch_size)
    np.random.shuffle(scales)
    boxes = np.zeros((len(scales), 4))
    for i, scale in enumerate(list(scales)):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]
    box_ind = np.arange(0, batch_size)
    return tf.image.crop_and_resize(
        x, boxes=boxes, box_indices=box_ind, crop_size=crop_size
    )


def clip(x: tf.Tensor) -> tf.Tensor:
    return tf.clip_by_value(x, 0, 1)


def augment(dataset: Dataset, batch_size=16, input_shape=(224, 224, 3)):
    for f in [
        # flip,
        color(input_shape[-1]),
        # rotate,
        partial(zoom, batch_size=batch_size, crop_size=input_shape[:-1]),
        # clip
    ]:
        dataset = dataset.map(
            map_func=lambda x, y: tf.cond(
                pred=tf.random.uniform([], 0, 1) > 0.5,
                true_fn=lambda: (f(x), y),
                false_fn=lambda: (x, y),
            ),
            num_parallel_calls=AUTOTUNE,
        )
    return dataset


def augment_pair(
    dataset: Dataset,
    batch_size=16,
    input_shape=(224, 224, 3),
    mdl_ins=["input_source", "input_target"],
    mdl_outs=["preds", "preds_1", "aux_out"],
):
    for f in [
        # flip,
        color(input_shape[-1]),
        # rotate,
        partial(zoom, batch_size=batch_size, crop_size=input_shape[:-1]),
        # clip
    ]:
        dataset = dataset.map(
            map_func=lambda x, y: tf.cond(
                pred=tf.random.uniform([], 0, 1) > 0.5,
                true_fn=lambda: ({k: f(x[k]) for k in mdl_ins}, y),
                false_fn=lambda: (x, y),
            ),
            num_parallel_calls=AUTOTUNE,
        )
    return dataset
