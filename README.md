# MIV-head

Official code repository for the paper (https://arxiv.org/abs/2507.00401)

We provide the code here, and instructions of executing the code, to reproduce the main experimental results in our paper


## 1. Preparation of test datasets


### 1.1 Go to fscmiv_sourcecode/data/ and install requirements:

```
cd fscmiv_sourcecode/data/
pip install -r requirements.txt
```


### 1.2 Specify Meta-Datasets (MD) root path (META_DATASET_ROOT) and output path (OUTPUT_ROOT) in the file of "data_config.py"

Within data_config.py, set the following variables:
- META_DATASET_ROOT = '/your_meta-dataset_root':
- OUTPUT_ROOT = '/your_output_root_of_data_and_schema'

By default, all 17 test datasets will be used. Otherwise, modify the "testsets" variable in the same file specifying datasets seperated by space


### 1.3 Retrieve raw data of test datasets from MD repository (TFRecords), saved within each data folder under OUTPUT_ROOT specified in (1.2) above

All raw data (images) are saved as .npy files and in their original resolutions, after running the following command: 

```
python3 records2npy.py
```


### 1.4 Extract sampling schemas from MD, saved under OUTPUT_ROOT specified in (1.2) above

All extracted schemas are saved as files named "XXX_standard.json" where XXX is a test-dataset name, e.g. OUTPUT_ROOT/omniglot_standard.json, after running the following command:

```
python3 dump_test_episodes.py
```


### 1.5 Create a fixed set of 600 random tasks from each test dataset

This step is to create the same set of tasks used to evaluate all algorithms, so that a paired t-test can be conducted to properly compare them. 
If this step is skipped, a new, different set of testing tasks (600 per each dataset) would be generated every time an evaluation is performed.

After running the following command, a fixed set of tasks will be created by randomly sampling based on the schemas extracted by (1.4) above.
By default, all 17 test datasets will be used to generate tasks saved under the folder "OUTPUT_ROOT/sampled".
If deviation from the default settings is required, one can modify the "SAMPLED_EPISODES" and "testsets" variables within the file of "data_config.py" accordingly
 
```
python3 create_fixed_episodes.py
```



## 2. Evaluations

By default, all evaluations use the data configurations specified in (1) above. This is our recommended setting. If different data configurations are needed, one should override the default value of related arguments in commands described below

By default, results of all evaluations are stored inside the folder "fscmiv_sourcecode/results/", including (i) a numpy array of accuracies for all testing tasks and (ii) .log file summarizing performance based on each test dataset. If no results need to be saved (only output to the stdout), use the argument "--no-npresults"


### 2.1 Go to fscmiv_sourcecode/ and install requirements:

```
cd fscmiv_sourcecode/
pip install -r requirements.txt
```


### 2.2 Download pretrained backbone model weights, saved in specified folders in "config_args.py"

Specify the default values of the following arguments, which indicate the storage paths of different pretrained backbone model weights:

- "dinovit_path": Path to store DINO-ViT pretrained weights that can be downloaded from [DINO repository](https://github.com/facebookresearch/dino)
- "dinornet_path": Path to store DINO-ResNet50 pretrained weights that can be downloaded from [DINO repository](https://github.com/facebookresearch/dino)
- "deit_path": Path to store DeiT(ViT) pretrained weights that can be downloaded from [DeiT repository](https://github.com/facebookresearch/deit/blob/main/README_deit.md)
- "sdlrnet_path": Path to store SDL-ResNet18 pretrained weights that can be downloaded from [URL repository](https://github.com/VICO-UoE/URL)
- "torch_hub": Path to store ResNets pretrained weights from Pytorch Hub that should be automatically downloaded when running our code in (2.2) below
- "hf_hub": Path to local cache of huggingface-hub. This argument is needed if the following backbones are used: CLIP(vit_l14clip), RegNet(cnn_regnet), DenseNet(cnn_densenet).
- "swin_path": Path to store Swin Transformer pretrained weights and config(.yaml) file. This argument is needed if the "swint_simmim" backbone is used.


### 2.2 Evaluate each algorithm

For each command described below, to get its full list of arguments and the explanations, see "config_args.py" or use '--help', e.g.,

```
python3 XXXXX.py --help
```

During the evaluation process, interim results will be printed to stdout every 10 tasks based on the "--print-freq" argument (default is 10). By default, the evaluation process will use the test data under OUTPUT_ROOT specified in the file of "data_config.py" (see Step 1.2 above). To override this option, one should use "--root" and "--sampled_path" arguments (both need to be specified) when running each algorithm.

After evaluations, one may collect all results under the folder "fscmiv_sourcecode/results/", which should be similar to the results reported in our paper. 


#### 2.2.1 MIV-head

To reproduce the results of the MIV-head in our paper, run the following command with the desired "--backbone" argument (default is "resnet50", the off-the-shelf backbone from pyTorch). 
For the "vit_dino" backbones, one may additionally specify "--patch_size" argument if required (default is 16).

```
python3 mivhead.py --backbone=vit_dino --patch_size=8 &> interim_results.log
```


#### 2.2.2 TSA

To reproduce the results of TSA in our paper, run the following command with the desired "--tsa_backbone" argument (default is "resnet50", the off-the-shelf backbone from pyTorch),

```
python3 tsa.py --tsa_backbone=dino_resnet50 &> interim_results.log
```


#### 2.2.3 eTT

To reproduce the results of eTT in our paper, run the following command with the desired "--ett_backbone" argument (default is "vit_dino", the off-the-shelf ViT backbone pretrained by DINO),

```
python3 ett.py --ett_backbone=vit_deit &> interim_results.log
```


#### 2.2.4 FiT-head

To reproduce the results of FiT-head in our paper, run the following command with the desired "--backbone" argument (shared with the same MIV-head argument),

```
python3 fithead.py --backbone=vit_dino &> interim_results.log
```


### 2.3 Collect measures of adaptation cost

To calculate and print out "training GFLOPS (per task)" of the MIV-head, TSA and eTT, use the argument "--gflops" (default is False). For example, running the following command provides GFLOPS information (without "accuracy" and "training time-duration"),

```
python3 mivhead.py --backbone=vit_dino --patch_size=16 --gflops &> gflops_only.log
```

Another measure of adaptation cost, "time duration of training (per task)" should already be printed along with the accuracy, when no "--gflops" argument is specified (by default).



## 3. Citation
If you find this repository useful, please cite [our paper](https://arxiv.org/abs/2507.00401)

TODO: bibtex


## 4. Acknowledgement

Part of our code is based on repositories of [TSA](https://github.com/VICO-UoE/URL), [eTT](https://github.com/chmxu/eTT_TMLR2022), [FiT](https://github.com/cambridge-mlg/fit), [FES](https://github.com/hongyujerrywang/featureextractorstacking)
