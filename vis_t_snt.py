# random select samples to form t-sne
from posixpath import split
import shutil
import os
from PIL import Image
full='occ_lfw_2.0'
dir_list = os.listdir(full)
for dir in dir_list:
  full_dest=os.path.join(full,dir)
  file_list=os.listdir(full_dest)
  if len(file_list)>19:
    print(dir)
# import argparse
# import json

# import time
# import shutil
# import logging
# import numpy as np

# import torch
# import torch.utils.data
# import torch.optim
# import torchvision.transforms as transforms
# import torch.backends.cudnn as cudnn

# cudnn.benchmark = True

# import lib.core.utils as utils
# from lib.core.config import config
# from lib.core.config import update_config
# from lib.core.functions import train
# from lib.core.lfw_eval import eval as lfw_eval
# from lib.datasets.dataset import WebFace_Folder
# from lib.datasets.dataset import WebFace_LMDB
# from lib.datasets.dataset import LFW_Image, MFR2_Image,MLFW_Image
# from lib.models.fpn import LResNet50E_IR_Occ as LResNet50E_IR_FPN
# ####using t-sne########
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# from sklearn import datasets
# from sklearn import manifold
# ################################


# # setup random seed
# torch.manual_seed(0)
# np.random.seed(0)

# def parse_args():
#     parser = argparse.ArgumentParser(description='Pytorch End2End Occluded Face')
#     parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
#     args, rest = parser.parse_known_args()
#     update_config(args.cfg)

#     parser.add_argument('--gpus', help='gpus', type=str)
#     parser.add_argument('--workers', help='num of dataloader workers', type=int)
#     parser.add_argument('--binary_thres', help='thres for binary mask', type=float)
#     parser.add_argument('--soft_binary', help='whether use soft binary mask', type=int)
#     parser.add_argument('--batch_size', help='batch size', type=int)
#     parser.add_argument('--pattern', help='num of pattern', type=int)
#     parser.add_argument('--pretrained', help='whether use pretrained model', type=str)
#     parser.add_argument('--debug', help='whether debug', default=0, type=int)
#     parser.add_argument('--model', help=' model name', type=str)
#     parser.add_argument('--factor', help='factor of mask',  type=float)
#     parser.add_argument('--ratio', help='ratio of masked img for training', default=4, type=int)
#     args = parser.parse_args()

#     return args

# def reset_config(config, args):
#     if args.gpus:
#         config.TRAIN.GPUS = args.gpus
#     if args.workers:
#         config.TRAIN.WORKERS = args.workers
#     if args.model:
#         print('update model type')
#         config.TRAIN.MODEL = args.model
#     if args.pattern:
#         print('update pattern and num_mask')
#         config.TRAIN.PATTERN = args.pattern
#         config.TRAIN.NUM_MASK = len(utils.get_grids(*config.NETWORK.IMAGE_SIZE, args.pattern))
#     if args.batch_size:
#         print('update batch_size')
#         config.TRAIN.BATCH_SIZE = args.batch_size
#         config.TEST.BATCH_SIZE = args.batch_size
#     if args.pretrained =='No':
#         print('update pretrained')
#         config.NETWORK.PRETRAINED = ''
#     if args.factor: 
#         print('update factor')
#         config.NETWORK.FACTOR = args.factor
#     if args.binary_thres:
#         print('update binary_thres')
#         config.TRAIN.BINARY_THRES = args.binary_thres
#     if args.soft_binary:
#         print('update soft_binary')
#         config.TRAIN.SOFT_BINARY = args.soft_binary

# def main():
#     # --------------------------------------model----------------------------------------
#     args = parse_args()
#     reset_config(config, args)
#     os.environ['CUDA_VISIBLE_DEVICES'] = config.TRAIN.GPUS
#     gpus = [int(i) for i in config.TRAIN.GPUS.split(',')]
#     gpus = range(len(gpus))

#     logger, final_output_dir, tb_log_dir = utils.create_temp_logger()

#     logger.info('Configs: \n' + json.dumps(config, indent=4, sort_keys=True))
#     logger.info('Args: \n' + json.dumps(vars(args), indent=4, sort_keys=True))

#     test_transform = transforms.Compose([
#         transforms.ToTensor(),  # ranges [0, 255] -> [0.0,1.0]
#         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # range [0.0, 1.0] -> [-1.0,1.0]
         
#     ])

#     test_loader = torch.utils.data.DataLoader(
#         # MLFW_Image(config,test_transform),
#         LFW_Image(config, test_transform),
#         # MFR2_Image(config,test_transform),
#         # O_LFW_Image(config,test_transform),
#         batch_size=config.TEST.BATCH_SIZE*len(gpus), 
#         shuffle=config.TEST.SHUFFLE,
#         # num_workers=config.TEST.WORKERS, 
#         num_workers=2,
#         pin_memory=True)

#     model_root ='pretrained/'
#     model_list = [
#     # 'baseline_L9_epoch15.pth.tar'
#     # 'baseline_resnet15_epoch17.pth.tar'
#     'twoNet_resnet15_9450.pth.tar'
#     # 'twoNet_resnet15_cbam_95.pth.tar'
#     # 'twoNets-G-checkpoint_23_91.pth.tar'
#     # 'twonet_L9_epoch22.pth.tar'
#     # 'twonets_resnet15-G-P-sum.pth.tar',
#     # 'twonets-resnet15-P-G-concat.pth.tar'
#     # 'twonets-resnet15-G.pth.tar'
#     # 'twonet-resnet15-G.pth.tar'
#     ]
#     # model_list = [
#         # 'model_best_p5_w0.pth.tar',
#         # 'model_best_p5_w1.pth.tar',
#         # 'model_best_p5_occ.pth.tar'
#     # ]
#     for model_name in model_list:
         
#         c_model = LResNet50E_IR_FPN( common=True,att_type=None)
#         p_model = LResNet50E_IR_FPN( common=False,att_type=None)
#         c_model = torch.nn.DataParallel(c_model, device_ids=gpus).cuda()
#         p_model = torch.nn.DataParallel(p_model, device_ids=gpus).cuda()
#         model_path = os.path.join(model_root, model_name)
#         checkpoint = torch.load(model_path)
#         c_state_dict = checkpoint['c_state_dict']
#         p_state_dict = checkpoint['p_state_dict']

#         if isinstance(c_model, torch.nn.DataParallel):
#             c_model.module.load_state_dict(c_state_dict, strict=False)
#             p_model.module.load_state_dict(p_state_dict, strict=False)
#         else:
#             c_model.load_state_dict(c_state_dict, strict=False)
#             p_model.load_state_dict(p_state_dict, strict=False)
        
     

