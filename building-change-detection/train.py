import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from datetime import datetime
from utils import dataloader
from utils.metrics import Evaluator
from utils.tools import adjust_lr, AvgMeter, print_network, poly_lr
import argparse
import logging
from network.PUGNet import PUGNet
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
# ----------------------------------------------------------------------------------------------------------------------

def Train(train_loader, Model, Model_optimizer, epoch, Eva):
    Model.train()
    loss_record_seg = AvgMeter()
    loss_record_conf = AvgMeter()
    print('Learning Rate: {}'.format(Model_optimizer.param_groups[0]['lr']))

    for i, sample in enumerate(tqdm(train_loader), start=1):
        Model_optimizer.zero_grad()
        A, B, mask = sample['A'], sample['B'], sample['label']
        A = Variable(A)
        B = Variable(B)
        gts = Variable(mask)
        A = A.cuda()
        B = B.cuda()
        Y = gts.cuda()

        outs = Model(A, B)

        seg_loss = CE_Loss(outs[0], Y.long()) + CE_Loss(outs[1], Y.long())+\
               CE_Loss(outs[2], Y.long()) + CE_Loss(outs[3], Y.long())
        
        loss_conf = outs[4]
        loss = seg_loss + loss_conf

        pred = outs[3].data.cpu().numpy()
        
        gt_target = Y.cpu().numpy()
        building_pred = np.argmax(pred, axis=1)
        building_pred = building_pred.astype(np.uint8)
        Eva.add_batch(gt_target, building_pred)
       
        # 反向传播：分割模型的梯度更新
        loss.backward()  # retain_graph=True 确保后续的 `backward()` 不会删除计算图
        Model_optimizer.step()  # 更新分割模型的参数

        loss_record_seg.update(seg_loss.data, opt.batchsize)
        loss_record_conf.update(loss_conf.data, opt.batchsize)

        if i % 100 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Seg Loss: {:.4f}, Conf Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record_seg.show(), loss_record_conf.show()))
            logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Seg Loss: {:.4f}, Conf Loss: {:.4f}'.
                         format(epoch, opt.epoch, i, total_step, loss_record_seg.show(), loss_record_conf.show()))
            
        
    IoU = Eva.Intersection_over_Union()[1]
    F1 = Eva.F1()[1]
    print('Epoch [{:03d}/{:03d}], \n[Training] IoU: {:.4f}, F1: {:.4f}'.format(epoch, opt.epoch, IoU, F1))

    logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], IoU: {:.4f}, F1: {:.4f}'.format(epoch, opt.epoch, IoU, F1))

def Val(test_loader, Model,  epoch, Eva, save_path):
    global best_f1, best_epoch
    Model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(test_loader)):
            A, B, mask = sample['A'], sample['B'], sample['label']
            A = A.cuda()
            B = B.cuda()
            Y = mask.cuda()
            res = Model(A, B)[3]

            pred = res.data.cpu().numpy()
        
            gt_target = Y.cpu().numpy()
            building_pred = np.argmax(pred, axis=1)
            building_pred = building_pred.astype(np.uint8)
            
            # Add batch sample into evaluator
            Eva.add_batch(gt_target, building_pred)
    IoU = Eva.Intersection_over_Union()[1]
    F1 = Eva.F1()[1]

    print('Epoch [{:03d}/{:03d}], \n[Validing] IoU: {:.4f}, F1: {:.4f}'.format(epoch, opt.epoch, IoU, F1))
    logging.info('#Val#:Epoch:{} IoU:{} F1:{}'.format(epoch, IoU, F1))
    new_f1 = F1
    if new_f1 >= best_f1:
        best_f1 = new_f1
        best_epoch = epoch
        print('Best Model Iou :%.4f; F1 : %.4f; Best epoch : %d' % (IoU, F1, best_epoch))

        torch.save(Model.state_dict(), save_path + 'Seg_epoch_best.pth')

    logging.info('#TEST#:Epoch:{} F1:{} bestEpoch:{} bestF1:{}'.format(epoch, F1, best_epoch, best_f1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--data_name', type=str, default='LEVIR-CD',
                        help='the test rgb images root')
    parser.add_argument('--backbone', type=str, default='PVT-v2',
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str,
                            default='./output/PUGNet/')
    opt = parser.parse_args()

    save_path = opt.save_path + opt.data_name + '/' + opt.backbone + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # build models
    Model = PUGNet(backbone_name = opt.backbone, channel = 32)
    Model.cuda()
    Model_params = Model.parameters()
    Model_optimizer = torch.optim.Adam(Model_params, opt.lr)

    # set path
    if opt.data_name == 'LEVIR-CD':
        opt.train_root = './Data/Change_Detection/LEVIR-CD_cropped256/train/' 
        opt.val_root = './Data/Change_Detection/LEVIR-CD_cropped256/val/'
        palatte = [[0,0,0], [255,255,255]]
    elif opt.data_name == 'Google':
        opt.train_root = './Data/Change_Detection/Google-CD/train/' 
        opt.val_root = './Data/Change_Detection/Google-CD/val/'
        palatte = [[0,0,0], [255,255,255]]
    elif opt.data_name == 'WHU':
        opt.train_root = './Data/Change_Detection/WHU-CD/train/' 
        opt.val_root = './Data/Change_Detection/WHU-CD/val/'
        palatte = [[0,0,0], [255,255,255]]
    elif opt.data_name == 'SYSU':
        opt.train_root = './Data/Change_Detection/SYSU-CD/train/' 
        opt.val_root = './Data/Change_Detection/SYSU-CD/val/'
        palatte = [[0,0,0], [255,255,255]]
    elif opt.data_name == 'Lebedev':
        opt.train_root = './Data/Change_Detection/Lebedev/train/' 
        opt.val_root = './Data/Change_Detection/Lebedev/val/'
        palatte = [[0,0,0], [255,255,255]]
    elif opt.data_name == 'Lebedev':
        opt.train_root = './Data/Change_Detection/LEVIR-CD+/train/' 
        opt.val_root = './Data/Change_Detection/LEVIR-CD+/test/'
        palatte = [[0,0,0], [255,255,255]]


    train_loader = dataloader.get_loader(img_A_root = opt.train_root + 'A/', img_B_root = opt.train_root + 'B/', gt_root = opt.train_root + 'label/', trainsize = opt.trainsize, palatte = palatte, mode ='train', batchsize = opt.batchsize, mosaic_ratio=0.25, num_workers=2, shuffle=True, pin_memory=True)
    test_loader = dataloader.get_loader(img_A_root = opt.val_root + 'A/', img_B_root = opt.val_root + 'B/', gt_root = opt.val_root + 'label/', trainsize = opt.trainsize, palatte = palatte, mode ='val', batchsize = opt.batchsize, mosaic_ratio=0, num_workers=2, shuffle=False, pin_memory=True)
    total_step = len(train_loader)

    logging.basicConfig(filename=save_path+'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Train")
    logging.info("Config")
    logging.info('epoch:{}; lr:{}; batchsize:{}; trainsize:{}; save_path:{}'.
                format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, save_path))

    # loss function
    CE_Loss = torch.nn.CrossEntropyLoss().cuda()
    print("Let's go!")
    best_f1 = 0
    best_epoch = 0
    Eva_tr = Evaluator(2)
    Eva_val = Evaluator(2)
    for epoch in range(1, (opt.epoch+1)):
        Eva_tr.reset()
        Eva_val.reset()
        lr = adjust_lr(Model_optimizer, opt.lr, epoch, 0.1, opt.decay_epoch)
        Train(train_loader, Model, Model_optimizer, epoch, Eva_tr)
        Val(test_loader, Model, epoch, Eva_val, save_path)



