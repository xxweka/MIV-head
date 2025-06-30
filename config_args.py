import argparse
from data.data_config import OUTPUT_ROOT, SAMPLED_EPISODES, testsets, test_type


parser = argparse.ArgumentParser(description='Few-shot classification as MIV')

# storage folders of backbone pretrained weights
parser.add_argument('--dinovit_path', type=str, help="Path to store DINO-ViT pretrained weights",
                    default='/your_path_to_store_dino-vit_pretrained_weights'
                    )
parser.add_argument('--deit_path', type=str, help="Path to store DeiT(ViT) pretrained weights",
                    default='/your_path_to_store_deit_pretrained_weights'
                    )
parser.add_argument('--torch_hub', type=str, help="Path to store ResNets pretrained weights from torch-hub",
                    default='/your_path_of_torchhub_resnets_pretrained_weights'
                    )
parser.add_argument('--dinornet_path', type=str, help="Path to store DINO-ResNet50 pretrained weights",
                    default='/your_path_to_store_dino-rnet50_pretrained_weights'
                    )
parser.add_argument('--sdlrnet_path', type=str, help="Path to store SDL-ResNet18 pretrained weights",
                    default='/your_path_to_store_sdl-rnet18_pretrained_weights'
                    )
parser.add_argument('--swin_path', type=str, help='Path to store Swin-Transformer pretrained weights, config(.yaml) file',
                    default='/your_path_to_store_swin_pretrained_weights'
                    )
parser.add_argument('--hf_hub', type=str, help='Path to local cache of huggingface-hub', default='/cache_of_huggingface/hub')

# data/experiment args
parser.add_argument("--print-freq", default=10, type=int, metavar="N", help="Print frequency of interim (moving-average) results (default: 10)")
parser.add_argument('--gpuid', type=str, default='0', metavar='GPU', help="ID of GPU to be used (default: 0)")
parser.add_argument('--num_workers', type=int, default=16, metavar='NEPOCHS', help="Number of workers that pre-process images in parallel (default: 16)")
parser.add_argument('--root', type=str, default=OUTPUT_ROOT, metavar='TESTROOT', help="Root folder of datasets used for testing")
parser.add_argument('--test_data', type=str, default=testsets, metavar='TESTSETS', help="Test datasets seperated by space")
parser.add_argument("--test_type", default=test_type, type=str, choices=('standard', '1shot', '5shot'), help="V-way N-shot")
parser.add_argument('--sampled_path', type=str, default=SAMPLED_EPISODES, metavar='TESTSAMPLES', help="Root folder of sampled episodes")
parser.add_argument('--finetune_steps', type=int, default=40, metavar='ITER', help='number of iterations/steps for adaptation (default: 40)')
parser.add_argument('--no-npresults', dest='npresults', action='store_false', help="Not output accuracy as np_array with _summary.log (defaul: output)")
parser.add_argument('--gflops', dest='gflops', action='store_true', help="Count and print GFLOPS (per task) information (defaul: No GFLOPS information)")

# shared arguments by MIV-head and FiT-head
parser.add_argument('--backbone', default='resnet50', type=str, help="Backbones of MIV-head (default: resnet50)",
                    choices=('resnet50', 'dino_resnet50', 'resnet18', 'sdl_resnet18', 'resnet34',
                             'vit_dino', 'vit_deit', 'swint_simmim', 'vit_l14clip', 'cnn_regnet', 'cnn_densenet'))
parser.add_argument('--patch_size', type=int, default=16, metavar='PS', choices=(8, 16), help="Patch size of DINO-ViT backbones (default: 16)")
parser.add_argument("--vit_arch", default='deitsmall', type=str, choices=('deitsmall', 'vitbase'), help="Small or base backbones of ViT (default: small)")

# MIV-head
parser.add_argument("--weight-decay", default=0, type=float, metavar="W", help='weight decay (default: 0)')
parser.add_argument('--finetune_lr', type=float, default=0.3, metavar='LR', help='CAP learning rate (default: 0.3)')

# FiT-head
parser.add_argument("--fithead_classifier", default='lda', type=str, choices=('lda', 'qda', 'protonets'), help='Variations of FiT-head (default: LDA)')
parser.add_argument('--fithead_finetune_lr', type=float, default=0.0035, metavar='LR', help='FiT-head learning rate (default: 0.0035)')

# Baseline++/Baseline
parser.add_argument("--classifier", default='blpp', type=str, choices=('blpp', 'bl'), help='Variants of classifier: Baseline++(blpp) or Baseline(bl) (default: blpp)')
parser.add_argument('--bl_finetune_lr', type=float, default=0.03, metavar='LR', help='Baseline++/Baseline learning rate (default: 0.03)')

# TSA
parser.add_argument('--tsa_ad_type', type=str, choices=['residual', 'serial', 'none'], default='residual', metavar='TSA_AD_TYPE', help="adapter type (default: residual)")
parser.add_argument('--tsa_ad_form', type=str, choices=['matrix', 'vector', 'none'], default='matrix', metavar='TSA_AD_FORM', help="adapter form (default: matrix)")
parser.add_argument('--tsa_head_params', type=str, choices=['alpha', 'beta', 'alpha+beta'], default='alpha+beta', metavar='TSA_OPT', help="task adaptation option (default: alpha+beta)")
parser.add_argument('--tsa_init', type=str, choices=['random', 'eye'], default='eye', metavar='TSA_INIT', help="initialization for adapter (default: identity matrix)")
parser.add_argument('--tsa_finetune_lr', type=float, default=0.5, metavar='LR', help='TSA alpha learning rate (default: 0.5)')
parser.add_argument('--tsa_beta_lr', type=float, default=1, metavar='LR', help='TSA beta learning rate (default: 1)')
parser.add_argument('--tsa_backbone', default='resnet50', type=str,
                    choices=('resnet50', 'dino_resnet50', 'resnet18', 'sdl_resnet18', 'resnet34'), help="Backbones of TSA"
                    )

# eTT
parser.add_argument('--ett_finetune_lr', type=float, default=1e-3, metavar='LR', help='eTT learning rate (default: 1e-3)')
parser.add_argument('--ett_backbone', default='vit_dino', type=str, choices=('vit_dino', 'vit_deit'), help="Backbones of eTT")
# # ViT-base is OOM for eTT
# parser.add_argument("--ett_vit_arch", default='deitsmall', type=str, choices=('deitsmall', 'vitbase'), help="ViT-small or ViT-base backbones for eTT")


# args as Namespace
args = parser.parse_args()
# # args as dict
# args = vars(parser.parse_args())
