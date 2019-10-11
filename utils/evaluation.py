from pathlib import Path
from sklearn.metrics import classification_report
import tensorflow as tf
keras = tf.compat.v2.keras
Dataset = tf.compat.v2.data.Dataset
Model = tf.compat.v2.keras.Model

def evaluate(
    model:Model, 
    test_dataset:Dataset, 
    test_size:int, 
    batch_size:int,  
    # pred_file_path:Path,
    report_path:Path,
    verbose:int,
    labels=None,
):
    y_true, y_pred = [], []
    steps = test_size//batch_size

    if verbose:
        print('Performing predictions on test set')

    tf_session = tf.compat.v1.keras.backend.get_session()
    test_iter = test_dataset.make_one_shot_iterator()
    next_elem = test_iter.get_next()
    for _ in range(steps):
        result = tf_session.run([next_elem])
        image_batch, label_batch = result[0]
        pred_prob_batch = model.predict(image_batch, steps=1, verbose=verbose)
        pred_label_batch = pred_prob_batch.argmax(axis=1)
        y_pred.extend(pred_label_batch)
        y_true.extend(label_batch.argmax(axis=1))

    if verbose:
        print('Creating classification report {}'.format(report_path))
        
    with open(report_path, 'w') as file:
        file.write('# Classification Report\n')
        file.write(classification_report(y_true=y_true, y_pred=y_pred, labels=labels))