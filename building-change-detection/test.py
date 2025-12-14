import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import dataloader
import numpy as np
from PIL import Image
from utils.metrics import Evaluator
from tqdm import tqdm
import argparse
from network.PUGNet import PUGNet
def onehot_to_mask(semantic_map, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    #x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    semantic_map = np.uint8(colour_codes[semantic_map])
    return semantic_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--data_name', type=str, default='LEVIR',
                        help='the test rgb images root')
    parser.add_argument('--backbone', type=str, default='PVT-v2',
                        help='')
    parser.add_argument('--save_path', type=str,
                            default='./output/PUGNet/')
    opt = parser.parse_args()
    palatte = [[0,0,0], [255,255,255]]

    if opt.data_name == 'LEVIR':
        test_root = './Data/Change_Detection/LEVIR-CD_cropped256/test/'
    elif opt.data_name == 'SYSU':
        test_root = './Data/Change_Detection/SYSU-CD/test/'
    elif opt.data_name == 'Google':
        test_root = './Data/Change_Detection/Google-CD/test/'
    elif opt.data_name == 'Lebedev':
        test_root = './Data/Change_Detection/Lebedev/test/'
    elif opt.data_name == 'WHU':
        test_root = './Data/Change_Detection/WHU-CD/test/'
    elif opt.data_name == 'LEVIR-CD+':
        test_root = './Data/Change_Detection/LEVIR-CD+/test/'
    save_path = './results/' + opt.data_name + '/PUGNet/'  
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        Eva = Evaluator(2)
        net = PUGNet(backbone_name = opt.backbone, channel = 32).cuda()
        model_path = opt.save_path + opt.data_name + '/' + opt.backbone + '/Seg_epoch_best.pth'
        net.load_state_dict(torch.load(model_path))
        test_load = dataloader.get_loader(img_A_root = test_root + 'A/', img_B_root = test_root + 'B/',\
                                        gt_root = test_root + 'label/', trainsize = opt.trainsize,\
                                        palatte = palatte, mode ='test',\
                                        batchsize = opt.batchsize, mosaic_ratio=0, num_workers=4, shuffle=False, pin_memory=True)
    
        print("Start Testing!")
        test(test_load, net, Eva, save_path)
def test(test_load, net, Eva, save_path):
    net.train(False)
    net.eval()
    for i, (sample, filename) in enumerate(tqdm(test_load)):
        A, B, mask = sample['A'], sample['B'], sample['label']
        A = A.cuda()
        B = B.cuda()
        Y = mask.cuda()
        
        output = net(A,B)[3]
        pred = output.data.cpu().numpy()
        
        gt_target = Y.cpu().numpy()
        building_pred = np.argmax(pred, axis=1)
        building_pred = building_pred.astype(np.uint8)
        
        # Add batch sample into evaluator
        Eva.add_batch(gt_target, building_pred)
        for i in range(building_pred.shape[0]):
            probs_array = np.squeeze(building_pred[i])
            final_savepath = save_path + '/' + filename[i] + '.png'
            im = Image.fromarray((probs_array*255).astype(np.uint8))
            im.save(final_savepath)

    IoU = Eva.Intersection_over_Union()[1]
    F1 = Eva.F1()[1]
    Precision = Eva.Precision()[1]
    Recall = Eva.Recall()[1]
    print(' IoU: %.4f, F1: %.4f, Precision: %.4f, Recall: %.4f' % (IoU, F1, Precision, Recall))




if __name__ == '__main__':
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    main()