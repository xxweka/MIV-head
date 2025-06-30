from fsc_modeling import create_fixed_episodes
from data.data_config import OUTPUT_ROOT, SAMPLED_EPISODES, test_datasets, test_type

schemas = [dataset + '_' + test_type + '.json' for dataset in test_datasets]
create_fixed_episodes(schema_root=OUTPUT_ROOT, sampled_path=SAMPLED_EPISODES, sample_schemas=schemas)
