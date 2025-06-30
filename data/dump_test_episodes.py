from meta_dataset_reader import MetaDatasetEpisodeReader
import torch
from tqdm import tqdm
import json
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

from data_config import test_datasets, test_type, OUTPUT_ROOT, TEST_SIZE


trainsets = testsets = test_datasets
test_loader = MetaDatasetEpisodeReader('test', trainsets, trainsets, testsets, test_type=test_type)

config = tf.compat.v1.ConfigProto()
# config.gpu_options.visible_device_list = '7'
config.gpu_options.allow_growth = False  # True
with tf.compat.v1.Session(config=config) as session:
    # go over each test domain
    for dataset in testsets:
        print(dataset)
        tasks = {}
        for i in tqdm(range(TEST_SIZE)):
            tasks[i] = {}
            with torch.no_grad():
                sample = test_loader.get_test_task(session, dataset)

                # query set
                qclass_new = sample['target_labels']
                qclass_orig = sample['target_gt']
                quniq, qcnt = qclass_new.unique(return_counts=True)
                count_qmap = {int(u): int(qcnt[c]) for c, u in enumerate(quniq)}

                # support set
                sclass_new = sample['context_labels']
                # sclass_orig = sample['context_gt']
                unique, count = sclass_new.unique(return_counts=True)
                count_smap = {int(u): int(count[c]) for c, u in enumerate(unique)}

                for j, key in enumerate(qclass_new):
                    k = int(key)
                    tasks[i][k] = {}
                    tasks[i][k]['class'] = str(int(qclass_orig[j]))
                    tasks[i][k]['query'] = count_qmap[k]
                    tasks[i][k]['support'] = count_smap[k]
            print(i, count_qmap, count_smap)

        with open(OUTPUT_ROOT+'/'+dataset+'_'+test_type+'.json', 'w') as jf:
            json.dump(tasks, jf)
