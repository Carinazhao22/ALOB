import argparse
import json
import os
import time
import shutil
import logging
import numpy as np

import torch
import torch.utils.data
import torch.optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

import lib.core.utils as utils
from lib.core.config import config
from lib.core.config import update_config
from lib.core.functions import train
from lib.core.lfw_eval import eval as lfw_eval
from lib.datasets.dataset import WebFace_Folder
from lib.datasets.dataset import WebFace_LMDB
from lib.datasets.dataset import LFW_Image, MFR2_Image,MLFW_Image
from lib.models.fpn_custresnet18 import LResNet50E_IR_Occ as LResNet50E_IR_FPN
# from lib.models.fpn import LResNet50E_IR_Occ_2D
# from lib.models.fpn import LResNet50E_IR_Occ_FC
from lib.models.metrics import ArcMarginProduct
from lib.models.metrics import CosMarginProduct
from lib.models.metrics import SphereMarginProduct

# setup random seed
torch.manual_seed(0)
np.random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch End2End Occluded Face')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--binary_thres', help='thres for binary mask', type=float)
    parser.add_argument('--soft_binary', help='whether use soft binary mask', type=int)
    parser.add_argument('--batch_size', help='batch size', type=int)
    parser.add_argument('--pattern', help='num of pattern', type=int)
    parser.add_argument('--pretrained', help='whether use pretrained model', type=str)
    parser.add_argument('--debug', help='whether debug', default=0, type=int)
    parser.add_argument('--model', help=' model name', type=str)
    parser.add_argument('--factor', help='factor of mask',  type=float)
    parser.add_argument('--ratio', help='ratio of masked img for training', default=4, type=int)
    args = parser.parse_args()

    return args

def reset_config(config, args):
    if args.gpus:
        config.TRAIN.GPUS = args.gpus
    if args.workers:
        config.TRAIN.WORKERS = args.workers
    if args.model:
        print('update model type')
        config.TRAIN.MODEL = args.model
    if args.pattern:
        print('update pattern and num_mask')
        config.TRAIN.PATTERN = args.pattern
        config.TRAIN.NUM_MASK = len(utils.get_grids(*config.NETWORK.IMAGE_SIZE, args.pattern))
    if args.batch_size:
        print('update batch_size')
        config.TRAIN.BATCH_SIZE = args.batch_size
        config.TEST.BATCH_SIZE = args.batch_size
    if args.pretrained =='No':
        print('update pretrained')
        config.NETWORK.PRETRAINED = ''
    if args.factor: 
        print('update factor')
        config.NETWORK.FACTOR = args.factor
    if args.binary_thres:
        print('update binary_thres')
        config.TRAIN.BINARY_THRES = args.binary_thres
    if args.soft_binary:
        print('update soft_binary')
        config.TRAIN.SOFT_BINARY = args.soft_binary

def main():
    # --------------------------------------model----------------------------------------
    args = parse_args()
    reset_config(config, args)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.TRAIN.GPUS
    gpus = [int(i) for i in config.TRAIN.GPUS.split(',')]
    gpus = range(len(gpus))

    logger, final_output_dir, tb_log_dir = utils.create_temp_logger()

    logger.info('Configs: \n' + json.dumps(config, indent=4, sort_keys=True))
    logger.info('Args: \n' + json.dumps(vars(args), indent=4, sort_keys=True))

    test_transform = transforms.Compose([
        transforms.ToTensor(),  # ranges [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # range [0.0, 1.0] -> [-1.0,1.0]
         
    ])

    test_loader = torch.utils.data.DataLoader(
        # MLFW_Image(config,test_transform),
        LFW_Image(config, test_transform),
        # MFR2_Image(config,test_transform),
        # O_LFW_Image(config,test_transform),
        batch_size=config.TEST.BATCH_SIZE*len(gpus), 
        shuffle=config.TEST.SHUFFLE,
        # num_workers=config.TEST.WORKERS, 
        num_workers=2,
        pin_memory=True)

    model_root ='pretrained/'
    model_list = [
    # 'baseline_L9_epoch15.pth.tar'
    # 'baseline_resnet15_epoch17.pth.tar'
    'twoNet_resnet15_9450.pth.tar'
    # 'twoNet_resnet15_cbam_95.pth.tar'
    # 'twoNets-G-checkpoint_23_91.pth.tar'
    # 'twonet_L9_epoch22.pth.tar'
    # 'twonets_resnet15-G-P-sum.pth.tar',
    # 'twonets-resnet15-P-G-concat.pth.tar'
    # 'twonets-resnet15-G.pth.tar'
    # 'twonet-resnet15-G.pth.tar'
    ]
    # model_list = [
        # 'model_best_p5_w0.pth.tar',
        # 'model_best_p5_w1.pth.tar',
        # 'model_best_p5_occ.pth.tar'
    # ]
    for model_name in model_list:
         
        c_model = LResNet50E_IR_FPN( common=True,att_type=None)
        p_model = LResNet50E_IR_FPN( common=False,att_type='CBAM')
        c_model = torch.nn.DataParallel(c_model, device_ids=gpus).cuda()
        p_model = torch.nn.DataParallel(p_model, device_ids=gpus).cuda()
        model_path = os.path.join(model_root, model_name)
        checkpoint = torch.load(model_path)
        c_state_dict = checkpoint['c_state_dict']
        p_state_dict = checkpoint['p_state_dict']

        if isinstance(c_model, torch.nn.DataParallel):
            c_model.module.load_state_dict(c_state_dict, strict=False)
            p_model.module.load_state_dict(p_state_dict, strict=False)
        else:
            c_model.load_state_dict(c_state_dict, strict=False)
            p_model.load_state_dict(p_state_dict, strict=False)
        
        # lfw_eval(c_model,p_model,model_path, config, test_loader, 'temp', 0)
        # print(c_model)
        # visualize(c_model,'c_model',test_transform)
        # visualize(p_model,'p_model',test_transform)
        print(' ')
def t_nst_vis(c_model,p_model,test_transform):
    from PIL import Image
    from sklearn import manifold
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    with open("/content/drive/MyDrive/FROM/sample_20.txt", "r") as selected_class:
        lines = selected_class.readlines()
    idx, count =0
    features=[]
    target=[]
    for name in lines:
        if count>4:
            idx+=1
            count=0
        image=Image.open(name)
        image=test_transform(image)
        image=image.unsqueeze(0)
        image=image.to('cuda')
        c_features=c_model(image)
        p_features = p_model(image,p_features)
        features.append(p_features.tolist())
        target.append(idx)
        
    # dimensionality reduction using t-SNE
    tsne = manifold.TSNE(n_components=2, random_state=42)
    # fit and transform
    mnist_tr = tsne.fit_transform(features)
    # transformed_data is a 2D numpy array of shape (100, 2)
    # create dataframe
    cps_df = pd.DataFrame(columns=['CP1', 'CP2', 'target'],
                        data=np.column_stack((mnist_tr, 
                                                target)))
    # cast targets column to int
    cps_df.loc[:, 'target'] = cps_df.target.astype(int)
    cps_df.head()

    identity_map = {0:'Gray_Davis',
               1: 'Rudolph_Giuliani',
               2: 'Angelina_Jolie',
               3: 'Carlos_Menem',
               4: 'Igor_Ivanov'}
    # map targets to actual clothes for plotting
    cps_df.loc[:, 'target'] = cps_df.target.map(identity_map)
    grid = sns.FacetGrid(cps_df, hue="target", size=8)
    grid.map(plt.scatter, 'CP1', 'CP2').add_legend()


def visualize(model,name,test_transform):
    import torch
    import torch.nn as nn
    import torchvision
    from torchvision import models, transforms, utils
    from torch.autograd import Variable
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.misc
    from PIL import Image
    import json
    image=Image.open("test/Aaron_Eckhart_0001.jpg")
    image=test_transform(image)
    image=image.unsqueeze(0)
    image=image.to('cuda')
    # print(model)
     
    #Iterate thru all the layers of the model
    
    # %matplotlib inline
    # we will save the conv layer weights in this list
    model_weights =[]
    #we will save the 49 conv layers in this list
    conv_layers = []
    # get all the model children as list
    model_children = list(model.modules())
    # print(model_children[0])
    #counter to keep count of the conv layers
    counter = 0
    #append all the conv layers and their respective wights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter+=1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter+=1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(name)
    print(f"Total convolution layers: {counter}")
    print("conv_layers")


    outputs = []
    names = []
    for layer in conv_layers[0:]:
      try: 
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))
      except:
        print("downstream")

    print(len(outputs))
    #print feature_maps
    for feature_map in outputs:
        print(feature_map.shape)
    
    processed = []
    for i in range(11):
        feature_map=outputs[i]
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())
    for fm in processed:
        print(fm.shape)
    # print(processed[0])
    fig = plt.figure(figsize=(50, 70))
    for i in range(len(processed)):
        a = fig.add_subplot(4, 4, i+1)
        imgplot = plt.imshow(processed[i])
        a.axis("off")
        a.set_title(names[i].split('(')[0], fontsize=30)
    plt.savefig('feature_maps_'+name+'.jpg', bbox_inches='tight')
        

if __name__ == '__main__':
    main()
