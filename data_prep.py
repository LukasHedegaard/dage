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
import tensorflow as tf
import data_util as dut

DATA_URL = 'https://github.com/SSARCandy/DeepCORAL/raw/master/dataset/office31.tar.gz'    
IMG_SHAPE = [227, 227, 3] # [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH]
SOURCE_SAMPLES_PER_DATASET = { 'amazon': 20, 'dslr': 8, 'webcam': 8 }
TARGET_SAMPLES_PER_DATASET = { 'amazon': 3, 'dslr': 3, 'webcam': 3 }
DIRNAME = 'data'
<<<<<<< HEAD
TVT_SPLITS = {
    'test' : 0.2,
    'train' : 0.8 * 0.8,
    'validation' : 0.8 * 0.2
}
=======
>>>>>>> dd488fa5e7392cdc817790cd3df4e676f74b813d


def all_paths_exist(path_list):
    return reduce( lambda lsum, p: lsum and os.path.exists(p), path_list, True)


def delete_if_exists(path):
    try:
        os.remove(path)
    except OSError:
        pass


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_sample_label_and_path_by_split(source_dir_path):
    classes = sorted(os.listdir(source_dir_path))
    glob_sample_info = {k:[] for k in TVT_SPLITS.keys()} # to be filled

    for class_index, class_name in enumerate(classes):
        class_dir = os.path.join(source_dir_path, class_name)
        class_sample_paths = tf.gfile.Glob(os.path.join(class_dir, '*.jpg'))
        random.shuffle(class_sample_paths)
        num_total = len(class_sample_paths)

<<<<<<< HEAD
        class_splits = { name: round(num_total * fraction) for (name, fraction) in TVT_SPLITS.items() }
=======
        class_splits = { name: round(num_total * fraction) for (name, fraction) in SPLITS.items() }
>>>>>>> dd488fa5e7392cdc817790cd3df4e676f74b813d
        class_splits['validation'] += num_total - sum(class_splits.values()) # rounding may result in missing or additional samples
        
        for (split_name, split_num_samples) in class_splits.items():
            for _ in range(split_num_samples):
               glob_sample_info[split_name].append( (class_index,(class_sample_paths.pop())) )
    
    return glob_sample_info


def create_train_val_test_splits(source_dir_path, target_dir_path):
    target_paths =  { s : os.path.join(target_dir_path, s+'.tfrecords')  
                      for s in TVT_SPLITS.keys() 
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
        output_file_path = target_paths[split_name]
        ensure_dir(output_file_path)
        with dut.open_writer(output_file_path) as writer:
            for sample_class_index, sample_path in split_data_info:
                sample = dut.serialise_image( dut.read_image(sample_path, IMG_SHAPE), sample_class_index)
                writer.write(sample)
    
    # create summary file
    summary = {
        split_name : { 'num_samples': len(sample_infos) }
        for split_name, sample_infos in glob_sample_info.items()
    }
    summary['classes'] = classes
    summary['data_IMG_SHAPE'] = {'height': IMG_SHAPE[0], 'width': IMG_SHAPE[1], 'depth': IMG_SHAPE[2]}
    summary_path = os.path.join(target_dir_path, 'summary.json')
    delete_if_exists(summary_path)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)


def get_dir_image_paths(dir_path):
    return tf.gfile.Glob(os.path.join(dir_path, '*.jpg'))


def sample_split(input_arr, n_in_sample):
    ''' NB: input_arr is shuffled '''
    random.shuffle(input_arr)
    sampled, rest = input_arr[:n_in_sample], input_arr[n_in_sample:]
    return sampled, rest


def create_domain_adaption_splits(dataset_name, images_dir_path, output_dir_path, n_splits=5):
    ''' Create the domain adaption splits as described in ''Simultaneous deep transfer across domains and tasks'' by Tzeng et al. 
        For source domain, sample 20 examples per category for the Amazon domain, or 8 examples per category for the DSLR and Webcam domains. 
        For target domain, sample 3 examples
        The target samples not used as target samples are used for testing.
        An additional dataset of all source samples is needed for fine-tuning.
    '''
    print('Creating domain adaption splits for {}'.format(dataset_name))

    sampled_output_paths = {
        'source': [ os.path.join(output_dir_path, 'splits/{}/source.tfrecords'.format(s)) for s in range(5) ],
        'target': [ os.path.join(output_dir_path, 'splits/{}/target.tfrecords'.format(s)) for s in range(5) ],
        'test':   [ os.path.join(output_dir_path, 'splits/{}/test.tfrecords'.format(s))   for s in range(5) ]
    }
    all_tfr_path = os.path.join(output_dir_path, 'all.tfrecords')

    all_output_tfr_paths = [all_tfr_path, *[path for paths in sampled_output_paths.values() for path in paths]]
    if (all_paths_exist(all_output_tfr_paths)):
        print('All TFRecords in {} seems to be in place. Skipping conversion'.format(output_dir_path))
        return
    
    print('Deleting old TFRecords')
    for p in all_output_tfr_paths:
        delete_if_exists(p)

    class_names = sorted(os.listdir(images_dir_path))
    label_image_paths = {
        label: get_dir_image_paths(os.path.join(images_dir_path, class_name))
        for label, class_name in enumerate(class_names)
    }

    for n in range(n_splits):
        print('Generating TFRecords for split #{}'.format(n))
        sampled = {'source':[],'target':[],'test':[]}
        for l, _ in enumerate(class_names):
            source_paths, _          = sample_split(label_image_paths[l], SOURCE_SAMPLES_PER_DATASET[dataset_name]) 
            target_paths, test_paths = sample_split(label_image_paths[l], TARGET_SAMPLES_PER_DATASET[dataset_name])
            sampled['source'].append({'p':source_paths, 'l':l})
            sampled['target'].append({'p':target_paths, 'l':l})
            sampled['test'  ].append({'p':test_paths,   'l':l})

        for data_split_name, pl_nested_list in sampled.items():
            pl_flat = [ {'p':p, 'l':pln['l']} for pln in pl_nested_list for p in pln['p'] ]
            random.shuffle(pl_flat)
            output_file_path = sampled_output_paths[data_split_name][n]
            ensure_dir(output_file_path)
            with dut.open_writer(output_file_path) as writer:
                for pl in pl_flat:
                    example = dut.serialise_image(image=dut.read_image(pl['p'], IMG_SHAPE), label=pl['l'])
                    writer.write(example)

    print('Generating TFRecords for all data')
    all_image_pl = [ {'p':path, 'l':label} for label, cip in label_image_paths.items() for path in cip ]
    random.shuffle(all_image_pl)
    with dut.open_writer(all_tfr_path) as writer:
        for pl in all_image_pl:
            example = dut.serialise_image(image=dut.read_image(pl['p'], IMG_SHAPE), label=pl['l'])
            writer.write(example)

    print('Generating summary.json')
    summary = {
        'all': {'num_samples' : len(all_image_pl)},
        'splits': {
            'source': {'num_samples': len(class_names)*SOURCE_SAMPLES_PER_DATASET[dataset_name]},
            'target': {'num_samples': len(class_names)*TARGET_SAMPLES_PER_DATASET[dataset_name]},
            'test':   {'num_samples': len(all_image_pl)-len(class_names)*TARGET_SAMPLES_PER_DATASET[dataset_name]},
            'num_splits': n_splits,
        },
        'classes': class_names,
        'data_shape': {'height': IMG_SHAPE[0], 'width': IMG_SHAPE[1], 'depth': IMG_SHAPE[2]}
    }
    summary_path = os.path.join(output_dir_path, 'summary.json')
    delete_if_exists(summary_path)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)


def convert_all_to_tfr():
    dirname = os.path.dirname(__file__)
    dest_directory = os.path.join(dirname, DIRNAME+'/office31')
    dataset_names = ['amazon', 'webcam', 'dslr']
    for name in dataset_names:
        images_dir_path = os.path.join(dest_directory, name+'/images')
        output_dir_path = os.path.join(dest_directory, name)
        create_domain_adaption_splits(name, images_dir_path, output_dir_path)


def download_and_extract():
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
    download_and_extract()
    convert_all_to_tfr()
    print("Job's done!")

if __name__ == '__main__':
    main()
