#备注MCNET是两个通道

import argparse
import logging
import os
import random
import shutil
import sys
import time
from torch.nn import BCEWithLogitsLoss, MSELoss
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from dataset import LiverDataset, LiverDataset_smi,LiverDataset_txt
from mcnet import MCNet2d_v1
from networks.net_factory import net_factory
from utils import losses, metrics, ramps

from metrics_sur import IoU, Dice,acc,SE,SP
from utils.util import compute_sdf

parser = argparse.ArgumentParser()
parser.add_argument('--train_path_img', type=str, default="Skin_Cancer_dataset/Skin_Cancer_dataset/big_data/train/image",
                    help='dir')
parser.add_argument('--train_path_label', type=str, default="Skin_Cancer_dataset/Skin_Cancer_dataset/big_data/train/label",
                    help='dir')
parser.add_argument('--val_path_img', type=str, default="Skin_Cancer_dataset/Skin_Cancer_dataset/big_data/test/image",
                    help='dir')
parser.add_argument('--val_path_label', type=str, default="Skin_Cancer_dataset/Skin_Cancer_dataset/big_data/test/label",
                    help='dir')
parser.add_argument('--exp', type=str,
                    default='ACDC/Adversarial_Network', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='dyunet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--DAN_lr', type=float,  default=0.0001,
                    help='DAN learning rate')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument('--beta', type=float,  default=0.3,
                    help='balance factor to control regional and sdm loss')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--runname', type=str, default="skin_2018_519_mcet",
                    help='Number of labeled data')
parser.add_argument('--gpu', default=0, type=int,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--lamda', type=float, default=1, help='weight to balance all losses')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
args = parser.parse_args()
writer_train = SummaryWriter("run/"+args.runname+"/train")
writer_val = SummaryWriter("run/"+args.runname+"/val")
wirter_all = SummaryWriter("run/"+args.runname+"/all")
x_transforms = transforms.Compose([
    # transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


y_transforms = transforms.Compose([
    transforms.ToTensor()
])
device = torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")




def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
def sharpening(P):
    T = 1/args.temperature
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen


def train():
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    # max_iterations = args.max_iterations
    def create_model(ema=False):
        # Network definition
        model = MCNet2d_v1(3,2)
       
    
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    model = create_model().to(device)
    model = model.to(device)
    

    liver_dataset = LiverDataset_txt(args.train_path_img, args.train_path_label,"skin_2018_file/train_519.txt" ,
                                transform=x_transforms, target_transform=y_transforms) #0-多少张
    liver_dataset_unlabel = LiverDataset_txt(args.train_path_img, args.train_path_label, "skin_2018_file/train_2075.txt",
                                transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloaders_unlabel = DataLoader(liver_dataset_unlabel, batch_size=batch_size, shuffle=True, num_workers=0)

    model.train()

    optimizer = optim.Adam(model.parameters(),lr=base_lr, betas=(0.9, 0.99))
    #optimizer = optim.SGD(model.parameters(), lr=0.01,
    #                     momentum=0.9, weight_decay=0.0001)
    
    ce_loss = CrossEntropyLoss()
    consistency_criterion = losses.mse_loss
    dice_loss = losses.DiceLoss(n_classes=num_classes)

    iter_num = 0
    train_iter =0
    epoch_loss = 0
    # max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = 300
    # n_unlable = len(dataloaders_unlabel.dataset)
    n_labeled = len(dataloaders.dataset)
    total_step = (n_labeled - 1) // dataloaders.batch_size + 1
    # total_step = (n_unlable - 1) // dataloaders_unlabel.batch_size + 1
    # total_step = 2
    labeled_bs = batch_size
    for epoch_num in range(iterator):
        print('Epoch {}/{}'.format(epoch_num, iterator - 1))
        print('-' * 10)
        model.train()
        train_iter += 1
        epoch_loss = 0
        Iou2 = 0
        dice2 = 0
        for iter_i in range(total_step): 
            labeled_train_iter = iter(dataloaders)
            volume_batch_labeled, label_batch_labeled = labeled_train_iter.next()
            volume_batch_labeled, label_batch_labeled = volume_batch_labeled.to(device), label_batch_labeled.to(device)

            
            unlabeled_train_iter = iter(dataloaders_unlabel)
            unlabeled_volume_batch,targets_u = unlabeled_train_iter.next()
            unlabeled_volume_batch = unlabeled_volume_batch.to(device)



            volume_batch = torch.cat([volume_batch_labeled,unlabeled_volume_batch],0)
            label_batch = torch.cat([label_batch_labeled,label_batch_labeled],0)

            model.train()
            outputs = model(volume_batch)
            num_outputs = len(outputs)

            y_ori = torch.zeros((num_outputs,) + outputs[0].shape)
            y_pseudo_label = torch.zeros((num_outputs,) + outputs[0].shape)

            loss_seg = 0
            loss_seg_dice = 0 
            for idx in range(num_outputs):
                y = outputs[idx][:labeled_bs]
                y_prob = F.softmax(y, dim=1)
                loss_seg += ce_loss(y, label_batch[:labeled_bs].long().squeeze(1))
                loss_seg_dice += dice_loss(y_prob, label_batch[:labeled_bs].float())

                y_all = outputs[idx]
                y_prob_all = F.softmax(y_all, dim=1)
                y_ori[idx] = y_prob_all
                y_pseudo_label[idx] = sharpening(y_prob_all)

            loss_consist = 0
            for i in range(num_outputs):
                for j in range(num_outputs):
                    if i != j:
                        loss_consist += consistency_criterion(y_ori[i], y_pseudo_label[j])
            
            iter_num = iter_num + 1
            consistency_weight = get_current_consistency_weight(iter_num//150)
            
            loss = args.lamda * loss_seg_dice + consistency_weight * loss_consist
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            outputs_soft = torch.softmax(outputs[0][:labeled_bs], dim=1)
            outputs_max = torch.max(outputs_soft,dim = 1)[1]
            I2 = IoU(outputs_max.to(device),label_batch.to(device))
            D2 = Dice(outputs_max.to(device),label_batch.to(device))
            Iou2 += I2
            dice2 += D2
           
           
         
            iter_num = iter_i + epoch_num * total_step
            # print("weight:%0.3f"% consistency_weight)
            print("%d/%d,train_g_loss:%0.3f, train_label_dice:%0.3f,train_label_iou:%0.3f" \
            % (iter_i, total_step, loss.item(),D2,I2))
        print("epoch %d loss:%0.3f" % (epoch_num, epoch_loss / total_step))
        print("train_mIou %0.3f" % (Iou2 /total_step))
        print("train_mdice %0.3f" % (dice2 /total_step))
        model.eval()


        batch_size1 = 1
        criterion1  = nn.CrossEntropyLoss()
        liver_dataset1 = LiverDataset_txt(args.val_path_img, args.val_path_label, "skin_2018_file/test_100.txt",
                                transform=x_transforms, target_transform=y_transforms)
    
        dataloaders1 = DataLoader(liver_dataset1, batch_size = batch_size1, shuffle=True, num_workers=2)
        with torch.no_grad():
            test_mloss,test_miou,test_mdice,test_macc,test_se,test_sp = test_model(model, criterion1,  dataloaders1)

        

        if test_mdice > best_performance:
            best_performance = test_mdice
            folder = os.path.exists("./weight2/"+args.runname)
            if not folder:
                os.makedirs("./weight2/"+args.runname)
            torch.save(model.state_dict(), "./weight2/"+args.runname+"/weights_best_epoch_%d_dice_%f.pth"%(epoch_num,test_mdice))
            
    
        print("best_dice%0.3f"%best_performance)
        writer_val.add_scalar("val_iou", test_miou, train_iter)
        writer_val.add_scalar("val_loss", test_mloss, train_iter)
        writer_val.add_scalar("val_dice", test_mdice, train_iter)
        writer_val.add_scalar("val_acc", test_macc, train_iter)
        writer_val.add_scalar("val_se", test_se, train_iter)
        writer_val.add_scalar("val_sp", test_sp, train_iter)



        wirter_all.add_scalars("loss",{'train_loss':epoch_loss/total_step,'val_loss':test_mloss},train_iter)
        wirter_all.add_scalars("iou", {'train_iou2': ((Iou2))/ total_step,
                                      'val_iuo1': test_miou}, train_iter)
        wirter_all.add_scalars("dice", {'train_dice2': ((dice2))/total_step,
                                      'val_dice1': test_mdice}, train_iter)

               

        
    
def test_model(model, criterion,  dataload, num_epochs=1):
    with torch.no_grad():
        n = 0
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            dt_size = len(dataload.dataset)
            epoch_loss = 0
            step = 0
            Iou = 0
            dice=0
            Acc =0
            se = 0
            sp = 0
            n = n + 1
            total_step = ((dt_size - 1) // dataload.batch_size + 1)
            for x, y in dataload:
                step += 1
                inputs = x.to(device)
                labels = y.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs[0], labels.long().squeeze(1))
                epoch_loss += loss.item()
                outputs_soft = torch.softmax(outputs[0], dim=1)
                outputs = torch.max(outputs_soft,1)[1]
                Iou += IoU(outputs.to(device), labels.to(device))
                dice += Dice(outputs.to(device), labels.to(device))
                Acc += acc(outputs.to(device), labels.to(device))
                se += SE(outputs.to(device), labels.to(device))
                sp += SP(outputs.to(device), labels.to(device))
            print("epoch %d loss:%0.3f" % (epoch, epoch_loss / total_step))
            print("test_mIou %0.3f" % (Iou / total_step))
            print("test_mdice %0.3f" % (dice / total_step))
            print("test_macc %0.3f" % (Acc /total_step))
            print("test_SE %0.3f" % (se /total_step))
            print("test_SP %0.3f" % (sp /total_step))
        return (epoch_loss / total_step),(Iou /total_step),(dice / total_step),(Acc / total_step),(se / total_step),(sp / total_step)


if __name__ == "__main__":
  
    train()
