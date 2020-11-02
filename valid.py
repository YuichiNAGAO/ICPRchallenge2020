from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import shutil
import numpy as np

from torch.autograd import Variable
from torch.utils.data.dataset import Subset

from model import Net2
import torch
import torch.optim as optim
from tqdm import tqdm
import pdb

import matplotlib.pyplot as plt
from Mydataset import MyDataset2
from myutils import make_dir

class Counter():
    def __init__(self):
        self.T = 0
        self.F = 0

        self.all_list = []
        
    
    def count(self, gt, pre):
        if gt == pre:
            self.T += 1
        else:
            self.F += 1
        self.all_list.append([gt, pre])

def load_checkpoint(file_path, use_cuda=False):
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)
    model = Net2()
    model.load_state_dict(checkpoint['state_dict'])
    return model


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', action='store', type = str, default='../data')
    parser.add_argument('--train_type', action='store', type = str, default='part',
                        choices=['all', 'part'],
                        help='train with all train data or use part of train data as validation [default: all]')
    parser.add_argument('--val', type=int, default=4, metavar='N', 
                        choices=[-1,1,2,3,4,5,6,7,8,9],
                        help = 'which train folder is used as validation (not trained) [default: -1]')
    parser.add_argument('--loss_type', action='store', type = str, default = 'CE',
                        choices=['FL', 'CE'],
                        help='loss type (Focal loss/Cross entropy) [default: CE]')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train [default:200]')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    root_dir = args.root
    
    results_dir = os.path.join(root_dir, 'results')
    model=load_checkpoint(os.path.join(results_dir, "{}_{}_{}_{}.pth".format(args.train_type,args.val,args.loss_type,args.epochs)), use_cuda=use_cuda)
    
    if use_cuda:
        model.cuda()
    model.eval()
    
    filling_type = np.load(os.path.join(root_dir, 'audio2', 'filling_type.npy'))
    pouring_or_shaking = np.load(os.path.join(root_dir,  'audio2', 'pouring_or_shaking.npy'))
    folder_count=np.load(os.path.join(root_dir, 'audio2', 'folder_count.npy'), allow_pickle=True)
    folder_count_detail=np.load(os.path.join(root_dir, 'audio2', 'folder_count_detail.npy'), allow_pickle=True)

    label = filling_type * pouring_or_shaking


    train_indices = []
    val_indices = []
    
    mydataset = MyDataset2(root_dir, test=False)
    n_samples = len(mydataset)
    l = list(range(0, n_samples))
    total_num = 0
    for i, num in enumerate(folder_count):
        if i == args.val-1:
            val_indices += l[total_num:total_num+num]
            
        total_num += num
    
    
    extracted_type=filling_type[np.array(val_indices)]
    last_ind=np.cumsum(np.array(folder_count_detail[args.val-1]))-1
    type_for_file=extracted_type[last_ind]
    
    
    val_dataset =  Subset(mydataset, val_indices)
    
    count=0
    total_num = 0
    correct_num = 0
    for j,gt  in zip(folder_count_detail[args.val-1],type_for_file):
        val_sub_dataset =  Subset(val_dataset, list(range(count,count+j)))
        count+=j
        val_sub_loader  = torch.utils.data.DataLoader(val_sub_dataset,
                                                 batch_size=1, 
                                                 shuffle=False)
        count_pred=[0,0,0,0]
        count_target=[0,0,0,0]
        print("gt",gt)
            
        for idx, (audio, target) in enumerate(val_sub_loader):
            if use_cuda:
                audio = audio.cuda()
            with torch.no_grad():
                audio = Variable(audio)
                pre = model(audio)
            _,pre=torch.max(pre,1)
            print("pred:{} target:{}".format(pre.item(),target.item()))
            count_pred[pre.item()]+=1
            count_target[target.item()]+=1
        
        if  count_pred[1]>5 or count_pred[2]>5 or count_pred[3]>5:
            final_pred=count_pred[1:4].index(max(count_pred[1:4]))+1
        else:
            final_pred=0
        if  gt==final_pred:
            print("True")
            correct_num+=1
        else:
            print("True")
    print("Acc",correct_num/len(folder_count_detail[args.val-1]))
            
            
        
    
    