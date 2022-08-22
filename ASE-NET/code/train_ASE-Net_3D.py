import os
# import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet import VNet
from networks.discriminator import FC3DDiscriminator

from dataloaders import utils
from utils import ramps, losses, metrics
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from utils.util import compute_sdf
from test_util_1 import test_all_case
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='data/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='LA/DTC_with_consis_weight', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=1,
                    help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='maximum epoch number to train')
parser.add_argument('--D_lr', type=float,  default=1e-4,
                    help='maximum discriminator learning rate to train')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=16, help='random seed')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--consistency_weight', type=float,  default=0.1,
                    help='balance factor to control supervised loss and consistency loss')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--beta', type=float,  default=0.3,
                    help='balance factor to control regional and sdm loss')
parser.add_argument('--gamma', type=float,  default=0.5,
                    help='balance factor to control supervised and consistency loss')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="kl", help='consistency_type')
parser.add_argument('--with_cons', type=str,
                    default="without_cons", help='with or without consistency')
parser.add_argument('--consistency', type=float,
                    default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=40.0, help='consistency_rampup')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "weight_our/" + args.exp + \
    "_{}labels_beta_{}/".format(
        args.labelnum, args.beta)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False  # True #
    cudnn.deterministic = True  # False #
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


if __name__ == "__main__":
    device = torch.device("cuda:%s"%args.gpu if torch.cuda.is_available() else "cpu")
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('', snapshot_path + '/code',
    #                 shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        # Network definition
        net = VNet(n_channels=1, n_classes=num_classes-1,
                   normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    # model = create_model()
    model = create_model().to(device)
    # model = model.to(device)
    ema_model = create_model(ema=True).to(device)


    DAN = FC3DDiscriminator(num_classes=num_classes-1)
    DAN = DAN.to(device)

    DAN1 = FC3DDiscriminator(num_classes=num_classes-1)
    DAN1 = DAN1.to(device)
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',  # train/val split
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))

    labelnum = args.labelnum    # default 16
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, 80))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=0, pin_memory=True)

    model.train()
    ema_model.train()
    DAN_optimizer = optim.Adam(
        DAN.parameters(), lr=0.001, betas=(0.9, 0.99))
    DAN_optimizer1 = optim.Adam(
        DAN1.parameters(), lr=0.001, betas=(0.9, 0.99))
    # optimizer = optim.SGD(model.parameters(), lr=base_lr,
    #                       momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(
        model.parameters(), lr=0.001, betas=(0.9, 0.99))
    ce_loss = BCEWithLogitsLoss()
    mse_loss = MSELoss()
    best_dice = 0
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            noise = torch.clamp(torch.randn_like(
                volume_batch[labeled_bs:]) * 0.1, -0.2, 0.2)
            ema_inputs = volume_batch[labeled_bs:] + noise

            DAN_target = torch.tensor([1] * 1).to(device)
            DAN_target_unlabel = torch.tensor([0] * 1).to(device)
            
           


            model.eval()
            
            DAN1.train()
            DAN.train()
            
            with torch.no_grad():
                outputs_q = model(volume_batch.float()).detach()
                outputs_soft = torch.sigmoid(outputs_q)
                outputs_soft_labeled = outputs_soft[:labeled_bs]
                outputs_soft_unlabel = outputs_soft[labeled_bs:]


                output_noise = model(ema_inputs).detach()
                output_noise_soft = torch.sigmoid(output_noise)
                outputs_ulabel_te = ema_model(volume_batch[labeled_bs:]).detach()
                outputs_soft_unlabel_te = torch.sigmoid(outputs_ulabel_te)

            DAN_outputs = DAN(outputs_soft_labeled.float(), volume_batch[:labeled_bs])
            DAN_loss1 = F.cross_entropy(DAN_outputs, DAN_target.long())

            DAN_outputs_unlabel = DAN(outputs_soft_unlabel.float(), volume_batch[labeled_bs:])
            DAN_loss2 = F.cross_entropy(DAN_outputs_unlabel, DAN_target_unlabel.long())
            DAN_loss = (DAN_loss1 + DAN_loss2)*0.5
            DAN_optimizer.zero_grad()
            DAN_loss.backward()

            DAN_outputs_1 = DAN1(outputs_soft_unlabel_te.float(), volume_batch[labeled_bs:])
            DAN_loss1_1 = F.cross_entropy(DAN_outputs_1, DAN_target.long())
            
            DAN_outputs_unlabel_1 = DAN1(output_noise_soft.float(), volume_batch[labeled_bs:])
            DAN_loss2_1 = F.cross_entropy(DAN_outputs_unlabel_1, DAN_target_unlabel.long())

            DAN_loss_1 = (DAN_loss1_1 + DAN_loss2_1)*0.5
            DAN_optimizer1.zero_grad()
            DAN_loss_1.backward()

            DAN_optimizer.step()
            DAN_optimizer1.step()

            model.train()
            
            DAN1.eval()
            DAN.eval()




            outputs = model(volume_batch)
            outputs_soft_1 = torch.sigmoid(outputs)
            # calculate the loss
            ema_output = ema_model(ema_inputs)
            output_noise = model(ema_inputs)

            outputs_ulabel = outputs[labeled_bs:]
            outputs_soft_unlabel_dan = outputs_soft_1[labeled_bs:]
            output_noise_soft_dan = torch.sigmoid(output_noise)
            loss_lu = losses.softmax_mse_loss_three(outputs_ulabel,output_noise,ema_output)
            consistency_dist = torch.mean(loss_lu)
            loss_seg = ce_loss(
                outputs[:labeled_bs, 0, ...], label_batch[:labeled_bs].float())
            loss_seg_dice = losses.dice_loss(
                outputs_soft[:labeled_bs, 0, :, :, :], label_batch[:labeled_bs] == 1)
            
            DAN_outputs = DAN(outputs_soft_unlabel_dan, volume_batch[labeled_bs:])

            DAN_outputs1 = DAN1(output_noise_soft_dan, volume_batch[labeled_bs:])
            consistency_loss = F.cross_entropy(DAN_outputs, DAN_target.long())
            consistency_loss1 = F.cross_entropy(DAN_outputs1, DAN_target.long())
            
            supervised_loss = (loss_seg_dice + loss_seg)*0.5

            consistency_weight = get_current_consistency_weight(iter_num//150)

            loss = supervised_loss + consistency_weight*((consistency_loss+consistency_loss1)*0.5+consistency_dist)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            dc = metrics.dice(torch.argmax(
                outputs_soft[:labeled_bs], dim=1), label_batch[:labeled_bs])

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_dice', loss_seg_dice, iter_num)
           
            writer.add_scalar('loss/consistency_weight',
                              consistency_weight, iter_num)
            writer.add_scalar('loss/consistency_loss',
                              consistency_loss, iter_num)

            logging.info(
                'iteration %d : loss : %f,  loss_seg: %f, loss_dice: %f' %
                (iter_num, loss.item(), 
                 loss_seg.item(), loss_seg_dice.item()))
            # writer.add_scalar('loss/loss', loss, iter_num)
            # logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
            if iter_num % 50 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, iter_num)

                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Dis2Mask', grid_image, iter_num)

               
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/DistMap', grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_DistMap',
                                 grid_image, iter_num)
                
            # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            # if iter_num % 500 == 0:
            #     save_mode_path = os.path.join(
            #         snapshot_path, 'iter_' + str(iter_num) + '.pth')
            #     torch.save(model.state_dict(), save_mode_path)
            #     logging.info("save model to {}".format(save_mode_path))
           
            if iter_num >= max_iterations:
                break
            time1 = time.time()
        ##########test
            if iter_num >= 500 and iter_num % 200 == 0:
               
                model.eval()
                ema_model().eval()
                with open("data/2018LA_Seg_Training Set" + '/test.list', 'r') as f:
                    image_list = f.readlines()
                image_list = ["data/2018LA_Seg_Training Set" + "/" + item.replace('\n', '') + "/mri_norm2.h5" for item in
                    image_list]
                with torch.no_grad():
                    avg_metric = test_all_case(model, image_list, num_classes=num_classes,
                                        patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                                        save_result=False,
                                        metric_detail=0, nms=1)
                    avg_metric1 = test_all_case(ema_model, image_list, num_classes=num_classes,
                                        patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                                        save_result=False,
                                        metric_detail=0, nms=1)

                print("test_dice %0.3f" % avg_metric1[0])
                print("test_jc %0.3f" % avg_metric1[1])
                print("test_hd %0.3f" % avg_metric1[2])
                print("test_asd %0.3f" % avg_metric1[3])
                
                print("test_dice %0.3f" % avg_metric[0])
                print("test_jc %0.3f" % avg_metric[1])
                print("test_hd %0.3f" % avg_metric[2])
                print("test_asd %0.3f" % avg_metric[3])
                
                writer.add_scalar('test_acc/dice',
                                    avg_metric[0], iter_num)

                writer.add_scalar('test_acc/jc',
                                    avg_metric[1], iter_num)
                writer.add_scalar('test_acc/hd',
                                    avg_metric[2], iter_num)

                writer.add_scalar('test_acc/asd',
                                    avg_metric[3], iter_num)
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(avg_metric[0]) + '.pth')
                save_mode_path1 = os.path.join(
                    snapshot_path, 'iter_' + str(avg_metric1[0]) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                torch.save(ema_model.state_dict(), save_mode_path1)
                logging.info("save model to {}".format(save_mode_path))
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
