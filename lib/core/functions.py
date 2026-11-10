import os
import time
import logging
import numpy as np

import torch
from tensorboardX import SummaryWriter



logger = logging.getLogger(__name__)

    
def train(train_loader, c_model, p_model,c_classifier,p_classifier, criterion, c_optimizer, p_optimizer, epoch, tb_log_dir, config):
    c_model.train()
    p_model.train()
    time_curr = time.time()
    p_loss_display = 0.0
    c_loss_display = 0.0

#    3) Training loop
    for batch_idx, data in enumerate(train_loader):
        img, label, _ , imgPaths = data
        img, label = img.cuda(), label.cuda()
    
        
        # 3.1 forward pass: compute the prediction
        c_features = c_model(img)
        p_features = p_model(img,c_features)

        
        if config.TRAIN.MODE == 'Mask':
          
          cfmap, pfmap = c_features,p_features
          c_output = c_classifier(cfmap, label)
          c_loss = criterion(c_output, label)
          c_loss_display += c_loss.item()
          # classfiy id by its 512 face embedding
          p_output = p_classifier(pfmap, label)
          p_loss = criterion(p_output, label)
          p_loss_display += p_loss.item()
          loss=p_loss+ config.LOSS.WEIGHT_PRED * c_loss
          
          c_optimizer.zero_grad()
          p_optimizer.zero_grad()
          
          loss.backward()
          c_optimizer.step()
          p_optimizer.step()      
         
        else:
            raise ValueError('Unknown training mode!')

 
        iters = epoch * len(train_loader) + batch_idx

        if iters % config.TRAIN.PRINT_FREQ == 0 and iters != 0:

            output = p_output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()
            acc = np.mean((output == label).astype(int))

            time_used = time.time() - time_curr
            if batch_idx < config.TRAIN.PRINT_FREQ:
                num_freq = batch_idx + 1
            else:
                num_freq = config.TRAIN.PRINT_FREQ
            speed = num_freq / time_used
            p_loss_display /= num_freq
            c_loss_display /= num_freq

            INFO = ' c_Margin: {:.2f}, p_Scale: {:.2f},p_Margin: {:.2f}, p_Scale: {:.2f}'.format(
                c_classifier.module.m, c_classifier.module.s,p_classifier.module.m, p_classifier.module.s)
            logger.info(
                'Train Epoch: {} [{:03}/{} ({:.0f}%)]{:05}, cLoss: {:.6f},pLoss: {:.6f} Acc: {:.4f}, Elapsed time: {:.4f}s, Batches/s {:.4f}'.format(
                    epoch, batch_idx, len(
                        train_loader), 100. * batch_idx / len(train_loader),
                    iters, c_loss_display, p_loss_display, acc, time_used, speed) + INFO)

            with SummaryWriter(tb_log_dir) as sw:
                sw.add_scalar('P_TRAIN_LOSS', p_loss_display, iters)
                sw.add_scalar('C_TRAIN_LOSS', c_loss_display, iters)
                sw.add_scalar('TRAIN_ACC', acc, iters)

            time_curr = time.time()
            p_loss_display = 0.0
            c_loss_display = 0.0

# def train(train_loader, c_model,c_classifier, criterion, c_optimizer, epoch, tb_log_dir, config):
#     c_model.train()
#     time_curr = time.time()
#     c_loss_display = 0.0

# #    3) Training loop
#     for batch_idx, data in enumerate(train_loader):
#         img, label, _ , imgPaths = data
#         img, label = img.cuda(), label.cuda()
    
        
#         # 3.1 forward pass: compute the prediction
#         c_features = c_model(img)
#         # p_features = p_model(img,c_features)

        
#         if config.TRAIN.MODE == 'Mask':
          
#           cfmap = c_features 
#           c_output = c_classifier(cfmap, label)
#           c_loss = criterion(c_output, label)
#           c_loss_display += c_loss.item()
#           # classfiy id by its 512 face embedding
#           loss=c_loss 
          
#           c_optimizer.zero_grad()
           
          
#           loss.backward()
#           c_optimizer.step()    
         
#         else:
#             raise ValueError('Unknown training mode!')

 
#         iters = epoch * len(train_loader) + batch_idx

#         if iters % config.TRAIN.PRINT_FREQ == 0 and iters != 0:

#             output = c_output.data.cpu().numpy()
#             output = np.argmax(output, axis=1)
#             label = label.data.cpu().numpy()
#             acc = np.mean((output == label).astype(int))

#             time_used = time.time() - time_curr
#             if batch_idx < config.TRAIN.PRINT_FREQ:
#                 num_freq = batch_idx + 1
#             else:
#                 num_freq = config.TRAIN.PRINT_FREQ
#             speed = num_freq / time_used
#             c_loss_display /= num_freq

#             INFO = ' c_Margin: {:.2f}, c_Scale: {:.2f}'.format(
#                 c_classifier.module.m, c_classifier.module.s)
#             logger.info(
#                 'Train Epoch: {} [{:03}/{} ({:.0f}%)]{:05}, cLoss: {:.6f}, Acc: {:.4f}, Elapsed time: {:.4f}s, Batches/s {:.4f}'.format(
#                     epoch, batch_idx, len(
#                         train_loader), 100. * batch_idx / len(train_loader),
#                     iters, c_loss_display, acc, time_used, speed) + INFO)

#             with SummaryWriter(tb_log_dir) as sw:
#                 sw.add_scalar('C_TRAIN_LOSS', c_loss_display, iters)
#                 sw.add_scalar('TRAIN_ACC', acc, iters)

#             time_curr = time.time()
#             c_loss_display = 0.0

 