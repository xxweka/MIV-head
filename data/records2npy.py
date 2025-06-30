import os
import gin
import sys
# import torch
import pickle
from data_config import META_DATASET_ROOT, OUTPUT_ROOT, META_RECORDS_ROOT, test_datasets
sys.path.insert(0, os.path.abspath(META_DATASET_ROOT))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm
import learning_spec  #, sampling, config, pipeline
import dataset_spec as dataset_spec_lib


def decode_image(example_string):
    image_string = tf.parse_single_example(
        example_string,
        features={
            'image': tf.FixedLenFeature([], dtype=tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        })['image']
    image_decoded = tf.image.decode_image(image_string, channels=3)
    image_decoded.set_shape([None, None, 3])
    image = tf.cast(image_decoded, tf.uint8)
    return image


for dataset in test_datasets:
    if not os.path.exists(f'{OUTPUT_ROOT}/{dataset}'):
        os.makedirs(f'{OUTPUT_ROOT}/{dataset}')
        print(dataset, flush=True)
        dataset_dir = f'{META_RECORDS_ROOT}/{dataset}'
        dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_dir)
        # print(dataset_spec, flush=True)
        for class_i in dataset_spec.get_classes(
                learning_spec.Split.TEST
                # learning_spec.Split.TRAIN
        ):
            class_dataset = tf.data.TFRecordDataset(f'{dataset_dir}/{class_i}.tfrecords')
            os.makedirs(f'{OUTPUT_ROOT}/{dataset}/{class_i}')
            for i, record in tqdm(enumerate(class_dataset)):
                np.save(f'{OUTPUT_ROOT}/{dataset}/{class_i}/{i}', decode_image(record).numpy())
