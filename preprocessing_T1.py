from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from model import Net
from AudioProcessing import AudioProcessing

import numpy as np
import scipy.io.wavfile
import os
from tqdm import tqdm
import pandas as pd
from glob import glob
import csv
import pdb

import torch
import time
import random

"""
$ python preprocessing_T1.py --root D:\codes\dataset 
"""

def load_checkpoint(file_path, use_cuda=False):
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)
    model = Net()
    model.load_state_dict(checkpoint['state_dict'])
    return model


"""
Constant 
"""
MAX_VALUE=194.19187653405487
MIN_VALUE=-313.07119549054045
save_size=64

"""Setup"""
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', action='store', type = str, default='./data')

    parser.add_argument('--train_type', action='store', type = str, default='all',
                        choices=['all', 'part'],
                        help='please set the same string as you have put in training phase!! train with all train data or use part of train data as validation [default: all]')
    parser.add_argument('--val', type=int, default=-1, metavar='N', 
                        choices=[-1,1,2,3,4,5,6,7,8,9],
                        help = 'please set the same number as you have put in training phase!! which train folder is used as validation (not trained) [default: -1]')
    parser.add_argument('--loss_type', action='store', type = str, default = 'CE',
                        choices=['FL', 'CE'],
                        help='please set the same string as you have put in training phase!! loss type (Focal loss/Cross entropy) [default: CE]')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='please set the same number as you have put in training phase!! number of epochs to train [default:200]')
    args = parser.parse_args()
    
    start = time.time()

    
    use_cuda = torch.cuda.is_available()
    root_dir = args.root
    results_dir = os.path.join(root_dir, 'T2_results')
    model=load_checkpoint(os.path.join(results_dir, "T2_{}_{}_{}_{}.pth".format(args.train_type,args.val,args.loss_type,args.epochs)), use_cuda=use_cuda)
    
    if use_cuda:
        model.cuda()
    model.eval()

    T2_mid_dir = os.path.join(root_dir, 'T2_mid')
    T2_pred_dir = os.path.join(root_dir, 'T2_pred')
    os.makedirs(T2_mid_dir,exist_ok=True)
    os.makedirs(T2_pred_dir,exist_ok=True)
    fol_count = 0
    mov_count = [0]*9
    for folder_num in range (1,10):
        pth = os.path.join(root_dir,str(folder_num), 'audio')
        files = glob(pth + "/*")
        for file in sorted(files):
            datalist = []
            predlist = []
            sample_rate, signal = scipy.io.wavfile.read(file)
            ap = AudioProcessing(sample_rate,signal)
            mfcc = ap.calc_MFCC()
            mfcc_length=mfcc.shape[0]
            f_step=int(mfcc.shape[1]*0.25)
            f_length=mfcc.shape[1]
            save_mfcc_num=int(np.ceil(float(np.abs(mfcc_length - save_size)) / f_step))
            for i in range(save_mfcc_num):
                tmp_mfcc = mfcc[i*f_step:save_size+i*f_step,: ,:]
                tmp_mfcc= (tmp_mfcc-MIN_VALUE)/(MAX_VALUE-MIN_VALUE)
                tmp_mfcc=tmp_mfcc.transpose(2,0,1)
                audio=torch.from_numpy(tmp_mfcc.astype(np.float32))
                audio=torch.unsqueeze(audio, 0)
                if use_cuda:
                    audio = audio.cuda()   
                output, pred=model.before_lstm(audio)
                _,pred=torch.max(pred,1)
                datalist.append(output.to('cpu').detach().numpy().copy())
                predlist.append(pred.item())
            fol_count +=1
            datalist = np.squeeze(np.array(datalist))
            predlist = np.squeeze(np.array(predlist))
            mov_count[folder_num-1] += 1
            np.save(os.path.join(T2_mid_dir, "{0:06d}".format(fol_count)), datalist)
            np.save(os.path.join(T2_pred_dir, "{0:06d}".format(fol_count)), predlist)
    np.save(os.path.join(root_dir, 'mov_count'), np.array(mov_count))

    elapsed_time = time.time() - start
    print("elapsed_time:{}".format(elapsed_time) + "sec")