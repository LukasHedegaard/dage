import json
from math import ceil
from pathlib import Path

import tensorflow as tf
from sklearn.metrics import classification_report
import numpy as np

keras = tf.compat.v2.keras
Dataset = tf.compat.v2.data.Dataset
Model = tf.compat.v2.keras.Model


def evaluate_da_pair(
    model: Model,
    test_dataset: Dataset,
    test_size: int,
    batch_size: int,
    report_path: Path,
    verbose: int,
    target_names=None,
):
    return evaluate(
        model,
        test_dataset,
        test_size,
        batch_size,
        report_path,
        verbose,
        target_names,
        get_data_and_lbls=lambda batch: (batch[0], batch[1]["preds"]),
        get_preds=lambda batch: batch[0],
    )


def evaluate_da_pair_orig(
    model: Model,
    test_dataset: Dataset,
    test_size: int,
    batch_size: int,
    report_path: Path,
    verbose: int,
    target_names=None,
):
    return evaluate(
        model,
        test_dataset,
        test_size,
        batch_size,
        report_path,
        verbose,
        target_names,
        get_data_and_lbls=lambda batch: (
            [batch[0]["input_source"], batch[0]["input_target"]],
            batch[1]["preds"],
        ),
        get_preds=lambda batch: batch[0],
    )


def evaluate(
    model: Model,
    test_dataset: Dataset,
    test_size: int,
    batch_size: int,
    report_path: Path,
    verbose: int,
    target_names=None,
    get_data_and_lbls=lambda x: x,
    get_preds=lambda x: x,
):
    assert tf.executing_eagerly()
    y_true, y_pred = [], []
    steps = ceil(test_size / batch_size)

    target_names = [str(tn) for tn in target_names]

    if verbose:
        print("Performing predictions on test set")

    progbar = keras.utils.Progbar(steps)
    ds_iter = iter(test_dataset)
    for batch in ds_iter:
        data, lbls = get_data_and_lbls(batch)
        pred_probs = model.predict_on_batch(data)
        pred_probs = get_preds(pred_probs)
        y_pred.extend(tf.argmax(pred_probs, axis=1))
        y_true.extend(lbls.numpy().argmax(axis=1))
        progbar.add(1)

    if verbose:
        print("Creating classification report at {}".format(report_path))
        print(
            classification_report(
                y_true=y_true,
                y_pred=y_pred,
                target_names=target_names,
                digits=4,
                output_dict=False,
            )
        )

    cr = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        target_names=target_names,
        digits=4,
        output_dict=True,
    )

    with open(report_path, "w") as f:
        json.dump(cr, f, indent=4)

    return cr


def extract_features(
    model: Model,
    test_dataset: Dataset,
    test_size: int,
    batch_size: int,
    outputs_dir: Path,
    verbose: int,
    target_names=None,
    get_data_and_lbls=lambda x: x,
    get_preds=lambda x: x,
):
    assert tf.executing_eagerly()
    y_true, y_pred = [], []
    steps = ceil(test_size / batch_size)

    target_names = [str(tn) for tn in target_names]

    if verbose:
        print("Performing feature extraction on test set")

    progbar = keras.utils.Progbar(steps)
    ds_iter = iter(test_dataset)
    for batch in ds_iter:
        data, lbls = get_data_and_lbls(batch)
        preds = model.predict_on_batch(data)
        preds = get_preds(preds)
        y_pred.extend(preds)
        y_true.extend(lbls.numpy().argmax(axis=1))
        progbar.add(1)

    if verbose:
        print(f"Saving features to {str(outputs_dir)}")

    np.save(outputs_dir / "test_features.npy", y_pred)
    np.save(outputs_dir / "test_targets.npy", y_true)

    return {}
