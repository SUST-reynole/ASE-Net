import argparse

import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import LiverDataset_txt

from UNet_2Plus_dy import UNet_2Plus_dy
from discriminator import Discriminator
from utils import losses, ramps
from metrics_sur import IoU, Dice,acc,SE,SP
from lib import  transforms_for_noise
parser = argparse.ArgumentParser()
parser.add_argument('--train_path_img', type=str, default="Skin_Cancer_dataset/big_data/train/image",
                    help='dir')
parser.add_argument('--train_path_label', type=str, default="Skin_Cancer_dataset/big_data/train/label",
                    help='dir')
parser.add_argument('--val_path_img', type=str, default="Skin_Cancer_dataset/big_data/test/image",
                    help='dir')
parser.add_argument('--val_path_label', type=str, default="Skin_Cancer_dataset/big_data/test/label",
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

parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--runname', type=str, default="skin_2018_our_519_9.04",
                    help='Number of labeled data')
parser.add_argument('--gpu', default=0, type=int,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
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

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
def train():
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    # max_iterations = args.max_iterations
    def create_model(ema=False):
        # Network definition
        model = UNet_2Plus_dy(3,2)
    
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    model = create_model().to(device)
    model = model.to(device)
    ema_model = create_model(ema=True).to(device)
    DAN = Discriminator(num_classes=num_classes)
    DAN = DAN.to(device)

    DAN1 = Discriminator(num_classes=num_classes)
    DAN1 = DAN1.to(device)

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
    DAN_optimizer = optim.Adam(
        DAN.parameters(), lr=args.DAN_lr, betas=(0.9, 0.99))
    DAN_optimizer1 = optim.Adam(
        DAN1.parameters(), lr=args.DAN_lr, betas=(0.9, 0.99))
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

  

    iter_num = 0
    train_iter =0
    epoch_loss = 0
    # max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = 300
    n_unlable = len(dataloaders_unlabel.dataset)
    n_labeled = len(dataloaders.dataset)
    # total_step = (n_labeled - 1) // dataloaders.batch_size + 1
    # total_step = (n_unlable - 1) // dataloaders_unlabel.batch_size + 1
    total_step = 600
    for epoch_num in range(iterator):
        print('Epoch {}/{}'.format(epoch_num, iterator - 1))
        print('-' * 10)
        train_iter += 1
        epoch_loss = 0
        Iou2 = 0
        dice2 = 0
        for i in range(total_step): 
            labeled_train_iter = iter(dataloaders)
            volume_batch, label_batch = labeled_train_iter.next()
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)

            
            unlabeled_train_iter = iter(dataloaders_unlabel)
            unlabeled_volume_batch,targets_u = unlabeled_train_iter.next()
            unlabeled_volume_batch = unlabeled_volume_batch.to(device)
            #添加噪声之后的输入
            if  np.random.random() < 0.25:
                volume_batch = transforms_for_noise(volume_batch, 0.5)
            inputs_u2_noise = transforms_for_noise(unlabeled_volume_batch, 0.5)
            #inputs_u2_noise, rot_mask, flip_mask = transforms_for_rot(inputs_u2_noise1)#旋转
            #inputs_u2_noise, shift_mask, scale_mask_shift1 = transforms_input_for_shift(inputs_u2_noise,256)
            
            ema_output = ema_model(inputs_u2_noise)
            output_noise = model(inputs_u2_noise)
            with torch.no_grad():
                #ema_output = transforms_output_for_shift(ema_output,shift_mask, scale_mask_shift1, 256)
                #ema_output = transforms_back_rot(ema_output, rot_mask, flip_mask)

                #output_noise = transforms_output_for_shift(output_noise,shift_mask, scale_mask_shift1, 256)
                #output_noise = transforms_back_rot(output_noise, rot_mask, flip_mask)
           
           
                output_noise_soft = torch.softmax(output_noise, dim=1)


            DAN_target = torch.tensor([1] * args.batch_size).to(device)
            DAN_target_unlabel = torch.tensor([0] * args.batch_size).to(device)
            model.train()
            ema_model.train()
            DAN.eval()
            DAN1.eval()
           
            outputs = model(volume_batch.float())
            outputs_soft = torch.softmax(outputs, dim=1)
            outputs_ulabel = model(unlabeled_volume_batch.float())
            outputs_soft_unlabel = torch.softmax(outputs_ulabel, dim=1)
            # input_all = torch.cat([volume_batch,unlabeled_volume_batch],dim=0)
            # output_all = model(input_all)
            # output_all_softmax = torch.softmax(output_all, dim=1)
            # outputs = output_all[:batch_size]
            # outputs_ulabel = output_all[batch_size:]
            # outputs_soft = output_all_softmax[:batch_size]
            # outputs_soft_unlabel = output_all_softmax[batch_size:]

            loss_ce = ce_loss(outputs,label_batch.long().squeeze(1))
            loss_dice = dice_loss(outputs_soft.float(), label_batch.float())
            loss_lu = losses.softmax_mse_loss_three(outputs_ulabel,output_noise,ema_output)
            consistency_dist = torch.mean(loss_lu)
            supervised_loss = 0.5 * (loss_dice + loss_ce)
            
            #consistency_weight = get_current_consistency_weight(iter_num//150)
            consistency_weight = get_current_consistency_weight(epoch_num)
            DAN_outputs = DAN(outputs_soft_unlabel, unlabeled_volume_batch)

            DAN_outputs1 = DAN1(output_noise_soft, unlabeled_volume_batch)
            consistency_loss = F.cross_entropy(DAN_outputs, DAN_target.long())
            consistency_loss1 = F.cross_entropy(DAN_outputs1, DAN_target.long())
            #loss = supervised_loss + consistency_weight * (consistency_loss +   consistency_dist+ consistency_loss1)

           
            
            loss = supervised_loss + consistency_weight * (consistency_loss  + consistency_loss1 + consistency_dist)
            #loss = supervised_loss + lmd * (consistency_loss  + consistency_loss1)
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            
            DAN1.train()
            DAN.train()
            with torch.no_grad():
                outputs = model(volume_batch.float())
                outputs_soft = torch.softmax(outputs, dim=1)
                outputs_unlabel = model(unlabeled_volume_batch.float())
                outputs_soft_unlabel = torch.softmax(outputs_unlabel, dim=1)
                # input_all = torch.cat([volume_batch,unlabeled_volume_batch],dim=0)
                # output_all = model(input_all)
                # output_all_softmax = torch.softmax(output_all, dim=1)
                # outputs = output_all[:batch_size]
                # outputs_ulabel = output_all[batch_size:]
                # outputs_soft = output_all_softmax[:batch_size]
                # outputs_soft_unlabel = output_all_softmax[batch_size:]

            DAN_outputs = DAN(outputs_soft.float().detach(), volume_batch)
            DAN_loss1 = F.cross_entropy(DAN_outputs, DAN_target.long())
            DAN_outputs_unlabel = DAN(outputs_soft_unlabel.float().detach(), unlabeled_volume_batch)
            DAN_loss2 = F.cross_entropy(DAN_outputs_unlabel, DAN_target_unlabel.long())
            DAN_loss = (DAN_loss1 + DAN_loss2)*0.5
            DAN_optimizer.zero_grad()
            DAN_loss.backward()
            


            
            
           
            with torch.no_grad():
                # outputs_unlabel_stu = model(inputs_u2_noise.float())
                # outputs_soft_stu = torch.softmax(outputs_unlabel_stu, dim=1)

                output_noise = model(inputs_u2_noise)

                #output_noise = transforms_output_for_shift(output_noise,shift_mask, scale_mask_shift1, 256)
                #output_noise = transforms_back_rot(output_noise, rot_mask, flip_mask)
                output_noise_soft = torch.softmax(output_noise, dim=1)


                outputs_ulabel_te = ema_model(unlabeled_volume_batch)

                # outputs_ulabel_te = transforms_output_for_shift(outputs_ulabel_te,shift_mask, scale_mask_shift1, 256)
                # outputs_ulabel_te = transforms_back_rot(outputs_ulabel_te, rot_mask, flip_mask)
                # outputs_soft_unlabel_te = torch.softmax(outputs_ulabel_te, dim=1)

                # outputs_ulabel_te = ema_model(unlabeled_volume_batch.float())
                outputs_soft_unlabel_te = torch.softmax(outputs_ulabel_te, dim=1)

            DAN_outputs_1 = DAN1(outputs_soft_unlabel_te.float().detach(), unlabeled_volume_batch)
            DAN_loss1_1 = F.cross_entropy(DAN_outputs_1, DAN_target.long())
            
            DAN_outputs_unlabel_1 = DAN1(output_noise_soft.float().detach(), unlabeled_volume_batch)
            DAN_loss2_1 = F.cross_entropy(DAN_outputs_unlabel_1, DAN_target_unlabel.long())
            DAN_loss_1 = (DAN_loss1_1 + DAN_loss2_1)*0.5
            DAN_optimizer1.zero_grad()
            DAN_loss_1.backward()
            DAN_optimizer.step()
            DAN_optimizer1.step()


            outputs = torch.max(outputs_soft,dim = 1)[1]
            I2 = IoU(outputs.to(device),label_batch.to(device))
            D2 = Dice(outputs.to(device),label_batch.to(device))
            Iou2 += I2
            dice2 += D2
            iter_num = i + epoch_num * total_step
            # print("weight:%0.3f"% consistency_weight)
            print("%d/%d,train_g_loss:%0.3f,train_d_loss1:%0.3f ,train_d_loss2:%0.3f,train_mseloss:%0.3f, train_label_dice:%0.3f,train_label_iou:%0.3f" \
            % (i, total_step, loss.item(),DAN_loss.item(),DAN_loss_1.item(),consistency_dist.item(),D2,I2))
            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr_
            
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            
        print("epoch %d loss:%0.3f" % (epoch_num, epoch_loss / total_step))
        print("train_mIou %0.3f" % (Iou2 /total_step))
        print("train_mdice %0.3f" % (dice2 /total_step))

        #model.load_state_dict(torch.load('weight/dyunet_train_adversarial_network_2D_our_GAN_D1_D2_10%_11.19_512/weights_teacher.pth',map_location='cuda'))
        model.eval()
    
        ema_model.eval()
        batch_size1 = 1
        criterion1  = nn.CrossEntropyLoss()
        liver_dataset1 = LiverDataset_txt(args.val_path_img, args.val_path_label, "skin_2018_file/test_100.txt",
                                transform=x_transforms, target_transform=y_transforms)
      
        dataloaders1 = DataLoader(liver_dataset1, batch_size = batch_size1, shuffle=True, num_workers=2)
        with torch.no_grad():
            test_mloss,test_miou,test_mdice,test_macc,test_se,test_sp = test_model(model, criterion1,  dataloaders1)

            test_mloss_v2,test_miou_v2,test_mdice_v2,test_macc_v2,test_se_v2,test_sp_v2 = test_model(ema_model, criterion1,  dataloaders1)

        if test_mdice > best_performance:
            best_performance = test_mdice
            folder = os.path.exists("./weight/"+args.runname)
            if not folder:
                os.makedirs("./weight/"+args.runname+"/")
            torch.save(model.state_dict(), './weight/'+args.runname+'/weights_best_epoch:%d_dice:%f.pth'%(epoch_num,test_mdice))
            #torch.save(ema_model.state_dict(), './weight/'+args.runname+'/weights_teacher.pth' )
        if test_mdice_v2 > best_performance:
            best_performance = test_mdice_v2
            folder = os.path.exists("./weight/"+args.runname)
            if not folder:
                os.makedirs("./weight/"+args.runname+"/")
            #torch.save(model.state_dict(), './weight/'+args.runname+'/weights best.pth')
            torch.save(ema_model.state_dict(), './weight/'+args.runname+'/weights_best_epoch:%d_dice:%f.pth'%(epoch_num,test_mdice_v2))
        print("best_dice%0.3f"%best_performance)
        writer_val.add_scalar("val_iou", test_miou, train_iter)
        writer_val.add_scalar("val_loss", test_mloss, train_iter)
        writer_val.add_scalar("val_dice", test_mdice, train_iter)
        writer_val.add_scalar("val_acc", test_macc, train_iter)
        writer_val.add_scalar("val_se", test_se, train_iter)
        writer_val.add_scalar("val_sp", test_sp, train_iter)


        writer_val.add_scalar("val_iou_teacher", test_miou_v2, train_iter)
        writer_val.add_scalar("val_loss_teacher", test_mloss_v2, train_iter)
        writer_val.add_scalar("val_dice_teacher", test_mdice_v2, train_iter)
        writer_val.add_scalar("val_acc_teacher", test_macc_v2, train_iter)
        writer_val.add_scalar("val_se_teacher", test_se_v2, train_iter)
        writer_val.add_scalar("val_sp_teacher", test_sp_v2, train_iter)

        wirter_all.add_scalars("loss",{'train_loss':epoch_loss/total_step,'val_loss':test_mloss},train_iter)
        wirter_all.add_scalars("iou", {'train_iou2': ((Iou2))/ total_step,
                                      'val_iuo1': test_miou,'val_iuo2': test_miou_v2}, train_iter)
        wirter_all.add_scalars("dice", {'train_dice2': ((dice2))/total_step,
                                      'val_dice1': test_mdice,'val_dice2': test_mdice_v2}, train_iter)

               

        
    
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
                labels = torch.squeeze(labels,dim=1)
                labels = labels.type(torch.LongTensor).to(device)
                loss = criterion(outputs, labels)
                epoch_loss += loss.item()
                outputs = torch.max(outputs,1)[1]
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
