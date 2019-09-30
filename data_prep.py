# Based on data preprocessing in https://github.com/harshitbansal05/Deep-Coral-Tf

import os
import sys
import tarfile
import json
from six.moves import urllib
import numpy as np
from functools import reduce
import h5py
import random
from PIL import Image
import tensorflow as tf
import tfrecord_util as tfr

DATA_URL = 'https://github.com/SSARCandy/DeepCORAL/raw/master/dataset/office31.tar.gz'    
DIRNAME = 'data'
SPLITS = {
    "test" : 0.2,
    "train" : 0.8 * 0.8,
    "validation" : 0.8 * 0.2
}
IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH = 227, 227, 3


def read_image(image_path):
    image = Image.open(image_path)
    image = image.resize([IMG_HEIGHT, IMG_WIDTH])
    image = np.array(image).tobytes()
    return image 


def all_paths_exist(path_list):
    return reduce( lambda lsum, p: lsum and os.path.exists(p), path_list, True)


def delete_if_exists(path):
    try:
        os.remove(path)
    except OSError:
        pass


def get_sample_label_and_path_by_split(source_dir_path):
    classes = sorted(os.listdir(source_dir_path))
    glob_sample_info = {k:[] for k in SPLITS.keys()} # to be filled

    for class_index, class_name in enumerate(classes):
        class_dir = os.path.join(source_dir_path, class_name)
        class_sample_paths = tf.gfile.Glob(os.path.join(class_dir, '*.jpg'))
        random.shuffle(class_sample_paths)
        num_total = len(class_sample_paths)
        class_splits = { name: round(num_total * fraction) for (name, fraction) in SPLITS.items() }
        class_splits["validation"] += num_total - sum(class_splits.values()) # rounding may result in missing or additional samples
        
        for (split_name, split_num_samples) in class_splits.items():
            for _ in range(split_num_samples):
               glob_sample_info[split_name].append( (class_index,(class_sample_paths.pop())) )
    
    return glob_sample_info


def maybe_convert_dataset_to_split_tfr(source_dir_path, target_dir_path):
    target_paths =  { s : os.path.join(target_dir_path, s+'.tfrecords')  
                      for s in SPLITS.keys() 
                    }

    if (all_paths_exist(target_paths.values())):
        print('All FRecords in {} seems to be in place. Skipping conversion'.format(target_dir_path))
        return
    
    print('Converting {} to TFRecords...'.format(source_dir_path))

    for p in target_paths.values():
        delete_if_exists(p)

    classes = sorted(os.listdir(source_dir_path))

    glob_sample_info = get_sample_label_and_path_by_split(source_dir_path)

    # collect samples in tfr
    for split_name, split_data_info in glob_sample_info.items():
        with tfr.open_writer(target_paths[split_name]) as writer:
            for sample_class_index, sample_path in split_data_info:
                sample = tfr.serialise_image(read_image(sample_path), sample_class_index)
                writer.write(sample)
    
    # create summary file
    summary = {
        split_name : { "num_samples": len(sample_infos) }
        for split_name, sample_infos in glob_sample_info.items()
    }
    summary["classes"] = classes
    summary["data_shape"] = {"height": IMG_HEIGHT, "width": IMG_WIDTH, "depth": IMG_DEPTH}
    summary_path = os.path.join(target_dir_path, 'summary.json')
    delete_if_exists(summary_path)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)


def maybe_convert_all_to_tfr():
    dirname = os.path.dirname(__file__)
    dest_directory = os.path.join(dirname, DIRNAME+'/office31')
    dataset_names = ['amazon', 'webcam', 'dslr']
    for name in dataset_names:
        source_dir_path = os.path.join(dest_directory, name+'/images')
        target_dir_path = os.path.join(dest_directory, name)
        maybe_convert_dataset_to_split_tfr(source_dir_path, target_dir_path)


def maybe_download_and_extract():
    dirname = os.path.dirname(__file__)
    dest_directory = os.path.join(dirname, DIRNAME)

    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory) 

    filename = DATA_URL.split('/')[-1]
    extracted_filename = DATA_URL.split('/')[-1].split('.')[0]
    filepath = os.path.join(dest_directory, filename)
    extracted_filepath = os.path.join(dest_directory, extracted_filename)

    if not os.path.exists(filepath) and not os.path.exists(extracted_filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                    float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    extracted_dir_path = os.path.join(dest_directory, 'office31')

    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    if os.path.exists(filepath):
        os.remove(filepath)



def main():
    maybe_download_and_extract()
    maybe_convert_all_to_tfr()
    print("Job's done!")

if __name__ == '__main__':
    main()
