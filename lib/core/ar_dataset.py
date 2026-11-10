import os
import json
import time
import argparse
import numpy as np
from PIL import Image
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import sys
sys.path.append('./')

import lib.core.utils as utils
from lib.core.config import config
from lib.models.fpn import LResNet50E_IR_Occ as LResNet50E_IR_FPN
from lib.models.resnets import LResNet50E_IR_Occ
from lib.datasets.dataset import ARdataset

# codes for end2end evaluation

# def parse_arguments():
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument('--protocol', type=int, help='', default=2)
#     parser.add_argument('--gallery-root', type=str, help='', default='data/datasets/ARdata/images_112_96/gallery_single/')
#     parser.add_argument('--glass-root', type=str, help='',default='data/datasets/ARdata/images_112_96/prob_glass_all')
#     parser.add_argument('--scarf-root', type=str, help='',default='data/datasets/ARdata/images_112_96/prob_scarf_all')
#     return parser.parse_args()

# def PIL_loader(path):
#     try:
#         with open(path, 'rb') as f:
#             return Image.open(f).convert('RGB')
#     except IOError:
#         print('Cannot load image ' + path)

# def get_models_params_new():
#     models_root = 'pretrained/'
#     models_names = [#'mymodel1_p5_2022-07-11-16-25_0.9933_0.9602_0.5230.pth.tar',
#                     #'mymodel1_p5_final_checkpoint.pth.tar',
#                     # 'mymodel3_p5_finalcheckpoint.pth.tar',
#                     # 'mymodel3_p5_best_2022-08-03-17-15_0.9935_0.9568_0.6623.pth.tar',
#                     # 'mymodel3_p5_best.pth_08-02-00-37.tar' 
#                     '0.945_model_best_p5.pth.tar'
#                     ]
#     # models_names = ['model_p4_baseline_9938_9563.pth.tar',]
#     # models_names = ['baseline_p4_arcface_9917.pth.tar',
#                     # 'baseline_p4_cosface_9927.pth.tar']

#     models_params = []
#     for name in models_names:
#         model_path = os.path.join(models_root, name)
#         assert os.path.exists(model_path), 'invalid model name!'

#         checkpoint = torch.load(model_path)
#         state_dict = checkpoint['state_dict']

#         model_name = name.split('.')[0]
#         models_params.append((model_name, state_dict))
#     return models_params


# def get_feature(img, model, model_name, mask=None):
#     if isinstance(mask, torch.Tensor):
#         #expand mask to match the batch size of img
#         batch_size = img.shape[0]
#         mask = mask.unsqueeze(0)
#         mask = mask.repeat(batch_size, 1, 1, 1)

#     img = img.to('cuda')
#     fc_mask, mask, _, fc = model(img, mask)
#     fc, fc_mask = fc.to('cpu').squeeze(), fc_mask.to('cpu').squeeze()
#     mask = mask.to('cpu')

#     if ('baseline' not in model_name):
#         fc = fc_mask
#     fc1 = fc[0::2, :]
#     fc2 = fc[1::2, :]
#     fc = fc1 + fc2
#     fc = F.normalize(fc1)

#     m1 = mask[0::2]
#     m2 = mask[1::2]
#     mask = (m1 + m2) / 2.0
#     return fc.detach(), mask.detach()

# def cosine_similarity(f1, f2):
#     # compute cosine_similarity for 1-D tensor
#     f1 = F.normalize(f1, dim=0)
#     f2 = F.normalize(f2, dim=0)

#     similarity = torch.sum(f1*f2).item()

#     return similarity

# def main():
#     args = parse_arguments()
#     if args.protocol == 1:
#         args.gallery_root = 'data/datasets/ARdata/images_112_96/gallery/'

#     print('Args: \n' + json.dumps(vars(args), indent=4))

#     os.environ['CUDA_VISIBLE_DEVICES'] = config.TRAIN.GPUS
#     gpus = [int(i) for i in config.TRAIN.GPUS.split(',')]
#     gpus = range(len(gpus))

#     transform = transforms.Compose([
#         transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
#         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
#     ])

#     gallery_loader = torch.utils.data.DataLoader(
#         ARdataset(args.gallery_root, transform),
#         batch_size=config.TEST.BATCH_SIZE, 
#         shuffle=config.TEST.SHUFFLE,
#         num_workers=2, 
#         pin_memory=True)
#     gallery_length = len(gallery_loader.dataset)

#     glass_loader = torch.utils.data.DataLoader(
#         ARdataset(args.glass_root, transform),
#         batch_size=config.TEST.BATCH_SIZE, 
#         shuffle=config.TEST.SHUFFLE,
#         num_workers=2, 
#         pin_memory=True)
#     glass_length = len(glass_loader.dataset)
#     batch_len = len(glass_loader)

#     scarf_loader = torch.utils.data.DataLoader(
#         ARdataset(args.scarf_root, transform),
#         batch_size=config.TEST.BATCH_SIZE, 
#         shuffle=config.TEST.SHUFFLE,
#         num_workers=2, 
#         pin_memory=True)
#     scarf_length = len(scarf_loader.dataset)
#     assert glass_length == scarf_length

#     for models_params in get_models_params_new():
#         model_name, state_dict = models_params
#         pattern = int(model_name[model_name.find('p')+1])
#         num_mask = len(utils.get_grids(*config.NETWORK.IMAGE_SIZE, pattern))
#         model = LResNet50E_IR_FPN(num_mask=num_mask)
#         model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

#         model.module.load_state_dict(state_dict, strict=False)
#         model.eval()

#         features, masks, labels = preprocessing(glass_length, glass_loader, model, model_name)
#         identification(model, model_name, features, masks, labels, gallery_length, gallery_loader, 'glass')

#         features, masks, labels = preprocessing(scarf_length, scarf_loader, model, model_name)
#         identification(model, model_name, features, masks, labels, gallery_length, gallery_loader, 'scarf')

# def preprocessing(length, loader, model, model_name):
#     features = torch.zeros(length // 2, 512)
#     masks = torch.zeros(length // 2, 512, 7, 6)
#     labels = []
#     begin = end = 0
#     for batch_idx, (img, label) in enumerate(loader):
#         fc, mask = get_feature(img, model, model_name)

#         begin = end
#         end = begin + len(label[::2])
#         features[begin:end] = fc
#         masks[begin:end] = mask
#         labels += label[::2]
#     return features, masks, labels

# def match(fc, label, features_gallery, labels_gallery):
#     similarity_label = []
#     for fc_gallery, label_gallery in zip(features_gallery, labels_gallery):
#         similarity = cosine_similarity(fc, fc_gallery)
#         similarity_label.append((similarity, label_gallery))
#     similarity_label = sorted(similarity_label, key=lambda x:x[0], reverse=True)
#     largest_similarity = similarity_label[0][0]
#     label_match = similarity_label[0][1]

#     return int(label == label_match)

# def identification(model, model_name, features, masks, labels, gallery_length, gallery_loader, probe_name):

#     mask_avg = torch.mean(masks, dim=0)
#     features_gallery = torch.zeros(gallery_length // 2, 512)
#     labels_gallery = []
#     start = end = 0
#     for batch_idx, (img, label_gallery) in enumerate(gallery_loader):
#         fc_gallery, _ = get_feature(img, model, model_name)

#         begin = end
#         end = begin + len(label_gallery[::2])
#         features_gallery[begin:end] = fc_gallery
#         labels_gallery += label_gallery[::2]

#     length = len(labels)
#     correct = 0
#     for i, fc, mask, label in zip(range(length), features, masks, labels):
#         correct += match(fc, label, features_gallery, labels_gallery)
#     acc = float(correct) / length
#     time_cur = time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime())
#     print('{}, model:{}, probe set:{}, rank 1 accuracy:{}'.format(time_cur, model_name, probe_name, acc))

# if __name__ == '__main__':
#     main()

# codes for twonets evaluation
def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--protocol', type=int, help='', default=2)
    parser.add_argument('--gallery-root', type=str, help='', default='data/datasets/ARdata/images_112_96/gallery_single/')
    parser.add_argument('--glass-root', type=str, help='',default='data/datasets/ARdata/images_112_96/prob_glass_all')
    parser.add_argument('--scarf-root', type=str, help='',default='data/datasets/ARdata/images_112_96/prob_scarf_all')
    return parser.parse_args()

def PIL_loader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)

def get_models_params_new():
    models_root = 'output/WebFace/LResNet50E_IR_FPN_CosMargin-Mask/CASIA-112x96-LMDB-Mask-pattern_5-weight_1.0-lr_0.01-optim_sgd-pretrained_0_factor_1/'
    models_names = [
                    # 'twoNet_model_p5_best_9450.pth.tar'
                    # 'twoNet_p5_Att_model_best_95.pth.tar'
                    # 'twoNetAtt_p5_model_best20_95.pth.tar'
                    'checkpoint.pth.tar'
                    #'mymodel1_p5_2022-07-11-16-25_0.9933_0.9602_0.5230.pth.tar',
                    #'mymodel1_p5_final_checkpoint.pth.tar',
                    # 'mymodel3_p5_finalcheckpoint.pth.tar',
                    # 'mymodel3_p5_best_2022-08-03-17-15_0.9935_0.9568_0.6623.pth.tar',
                    # 'mymodel3_p5_best.pth_08-02-00-37.tar' 
                    ]
    # models_names = ['model_p4_baseline_9938_9563.pth.tar',]
    # models_names = ['baseline_p4_arcface_9917.pth.tar',
                    # 'baseline_p4_cosface_9927.pth.tar']

    models_params = []
    for name in models_names:
        model_path = os.path.join(models_root, name)
        assert os.path.exists(model_path), 'invalid model name!'

        checkpoint = torch.load(model_path)
        # state_dict = checkpoint['state_dict']
        c_state_dict = checkpoint['c_state_dict']
        p_state_dict = checkpoint['p_state_dict']

        model_name = name.split('.')[0]
        models_params.append((model_name, c_state_dict,p_state_dict))
    return models_params


def get_feature(img, c_model,p_model,model_name, mask=None):
# def get_feature(img, c_model,model_name, mask=None):
    if isinstance(mask, torch.Tensor):
        #expand mask to match the batch size of img
        batch_size = img.shape[0]
        mask = mask.unsqueeze(0)
        mask = mask.repeat(batch_size, 1, 1, 1)

    img = img.to('cuda')
    # fc_mask, mask, _, fc = c_model(img)
    c_features = c_model(img)
    p_featres=p_model(img,c_features)
 
    p_featres=p_featres.to('cpu').squeeze()
    # c_features=c_features.to('cpu').squeeze()

    # if ('baseline' not in model_name):
    #     fc = fc_mask
    fc1 = p_featres[0::2, :]
    fc2 = p_featres[1::2, :]
    # fc1 = c_features[0::2, :]
    # fc2 = c_features[1::2, :]
    fc = fc1 + fc2
    fc = F.normalize(fc1)

    # m1 = mask[0::2]
    # m2 = mask[1::2]
    # mask = (m1 + m2) / 2.0
    # return fc.detach(), mask.detach()
    return fc.detach()

def cosine_similarity(f1, f2):
    # compute cosine_similarity for 1-D tensor
    f1 = F.normalize(f1, dim=0)
    f2 = F.normalize(f2, dim=0)

    similarity = torch.sum(f1*f2).item()

    return similarity

def main():
    args = parse_arguments()
    if args.protocol == 1:
        args.gallery_root = 'data/datasets/ARdata/images_112_96/gallery/'

    print('Args: \n' + json.dumps(vars(args), indent=4))

    os.environ['CUDA_VISIBLE_DEVICES'] = config.TRAIN.GPUS
    gpus = [int(i) for i in config.TRAIN.GPUS.split(',')]
    gpus = range(len(gpus))

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    gallery_loader = torch.utils.data.DataLoader(
        ARdataset(args.gallery_root, transform),
        batch_size=config.TEST.BATCH_SIZE, 
        shuffle=config.TEST.SHUFFLE,
        num_workers=2, 
        pin_memory=True)
    gallery_length = len(gallery_loader.dataset)

    glass_loader = torch.utils.data.DataLoader(
        ARdataset(args.glass_root, transform),
        batch_size=config.TEST.BATCH_SIZE, 
        shuffle=config.TEST.SHUFFLE,
        num_workers=2, 
        pin_memory=True)
    glass_length = len(glass_loader.dataset)
    batch_len = len(glass_loader)

    scarf_loader = torch.utils.data.DataLoader(
        ARdataset(args.scarf_root, transform),
        batch_size=config.TEST.BATCH_SIZE, 
        shuffle=config.TEST.SHUFFLE,
        num_workers=2, 
        pin_memory=True)
    scarf_length = len(scarf_loader.dataset)
    assert glass_length == scarf_length

    for models_params in get_models_params_new():
        model_name, c_state_dict,p_state_dict = models_params
        # model_name, c_state_dict= models_params
        # pattern = int(model_name[model_name.find('p')+1])
        # num_mask = len(utils.get_grids(*config.NETWORK.IMAGE_SIZE, pattern))
        c_model = LResNet50E_IR_FPN(common=True)
        p_model = LResNet50E_IR_FPN(common=False)
        
        c_model = torch.nn.DataParallel(c_model, device_ids=gpus).cuda()
        p_model = torch.nn.DataParallel(p_model, device_ids=gpus).cuda()

        c_model.module.load_state_dict(c_state_dict, strict=False)
        p_model.module.load_state_dict(p_state_dict, strict=False)
        
        c_model.eval()
        p_model.eval()

        features, labels = preprocessing(glass_length, glass_loader, c_model, p_model,model_name)
        identification(c_model,p_model, model_name, features, labels, gallery_length, gallery_loader, 'glass')

        features, labels = preprocessing(scarf_length, scarf_loader, c_model,p_model,model_name)
        identification(c_model,p_model, model_name, features, labels, gallery_length, gallery_loader, 'scarf')
        # features, labels = preprocessing(glass_length, glass_loader, c_model,model_name)
        # identification(c_model,model_name, features, labels, gallery_length, gallery_loader, 'glass')

        # features, labels = preprocessing(scarf_length, scarf_loader, c_model,model_name)
        # identification(c_model, model_name, features, labels, gallery_length, gallery_loader, 'scarf')

def preprocessing(length, loader, c_model,p_model, model_name):
    features = torch.zeros(length // 2, 512)
    # masks = torch.zeros(length // 2, 512, 7, 6)
    labels = []
    begin = end = 0
    for batch_idx, (img, label) in enumerate(loader):
        # fc, mask 
        # print(length//2)
        # print(img.shape)
        # print(len(label))
        fmap= get_feature(img, c_model,p_model, model_name)
        # print(fmap.shape)
        # print(label[::2])
        # print(label)
        begin = end
        end = begin + len(label[::2])
        features[begin:end] = fmap
        # masks[begin:end] = mask
        labels += label[::2]
    return features,  labels

# def preprocessing(length, loader, c_model, model_name):
#     features = torch.zeros(length // 2, 512)
#     # masks = torch.zeros(length // 2, 512, 7, 6)
#     labels = []
#     begin = end = 0
#     for batch_idx, (img, label) in enumerate(loader):
#         # fc, mask 
#         # print(length//2)
#         # print(img.shape)
#         # print(len(label))
#         fmap= get_feature(img, c_model, model_name)
#         # print(fmap.shape)
#         # print(label[::2])
#         # print(label)
#         begin = end
#         end = begin + len(label[::2])
#         features[begin:end] = fmap
#         # masks[begin:end] = mask
#         labels += label[::2]
#     return features,  labels


def match(fc, label, features_gallery, labels_gallery):
    similarity_label = []
    for fc_gallery, label_gallery in zip(features_gallery, labels_gallery):
        similarity = cosine_similarity(fc, fc_gallery)
        similarity_label.append((similarity, label_gallery))
    similarity_label = sorted(similarity_label, key=lambda x:x[0], reverse=True)
    largest_similarity = similarity_label[0][0]
    label_match = similarity_label[0][1]

    return int(label == label_match)

# def identification(c_model,model_name, features, labels, gallery_length, gallery_loader, probe_name):

#     # mask_avg = torch.mean(masks, dim=0)
#     features_gallery = torch.zeros(gallery_length // 2, 512)
#     labels_gallery = []
#     start = end = 0
#     for batch_idx, (img, label_gallery) in enumerate(gallery_loader):
#         fc_gallery= get_feature(img, c_model,model_name)

#         begin = end
#         end = begin + len(label_gallery[::2])
#         features_gallery[begin:end] = fc_gallery
#         labels_gallery += label_gallery[::2]

#     length = len(labels)
#     correct = 0
#     for i, fc, label in zip(range(length), features, labels):
#         correct += match(fc, label, features_gallery, labels_gallery)
#     acc = float(correct) / length
#     time_cur = time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime())
#     print('{}, model:{}, probe set:{}, rank 1 accuracy:{}'.format(time_cur, model_name, probe_name, acc))


def identification(c_model,p_model, model_name, features, labels, gallery_length, gallery_loader, probe_name):

    # mask_avg = torch.mean(masks, dim=0)
    features_gallery = torch.zeros(gallery_length // 2, 512)
    labels_gallery = []
    start = end = 0
    for batch_idx, (img, label_gallery) in enumerate(gallery_loader):
        fc_gallery= get_feature(img, c_model,p_model,model_name)

        begin = end
        end = begin + len(label_gallery[::2])
        features_gallery[begin:end] = fc_gallery
        labels_gallery += label_gallery[::2]

    length = len(labels)
    correct = 0
    for i, fc, label in zip(range(length), features, labels):
        correct += match(fc, label, features_gallery, labels_gallery)
    acc = float(correct) / length
    time_cur = time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime())
    print('{}, model:{}, probe set:{}, rank 1 accuracy:{}'.format(time_cur, model_name, probe_name, acc))

if __name__ == '__main__':
    main()