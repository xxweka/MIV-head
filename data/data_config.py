# root paths
META_DATASET_ROOT = '/your_meta-datasets_root'
META_RECORDS_ROOT = META_DATASET_ROOT + '/records'
OUTPUT_ROOT = '/your_output_root_of_data_and_schema'
SAMPLED_EPISODES = OUTPUT_ROOT + '/sampled'

# datasets
testsets = 'omniglot cu_birds dtd fungi vgg_flower traffic_sign mscoco aircraft quickdraw mnist cifar10 cifar100 CropDisease EuroSAT ISIC ChestX Food101'
test_datasets = testsets.split(' ')
test_type = 'standard'  # ['standard', '1shot', '5shot']
TEST_SIZE = 600
