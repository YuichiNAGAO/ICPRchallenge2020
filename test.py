from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from model import Net
from audio_processing import AudioProcessing

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

"""setup"""
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', action='store', type = str, default='./data')
    parser.add_argument('--folder_num', action='store',type=int, nargs='*',default=[10, 11, 12],help='name of folder that you want to test')
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
    parser.add_argument('--target', action='store', type = str, default = 'private',
                        choices=['public', 'private'],
                        help='select the target dataset (public test set/private test set) [default: private]')
    args = parser.parse_args()
    
    start = time.time()

    #header = ['Object','Sequence','Container capacity [mL]','Container mass [g]','Filling type','Filling level [%]','Filling mass [g]']
    header = ['Container ID','Sequence','Fullness','Filling type', 'Container capacity [mL]']
    with open('./submission.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    
    use_cuda = torch.cuda.is_available()
    root_dir = args.root
    results_dir = os.path.join(root_dir, 'results')
    model=load_checkpoint(os.path.join(results_dir, "{}_{}_{}_{}.pth".format(args.train_type,args.val,args.loss_type,args.epochs)), use_cuda=use_cuda)
    
    if use_cuda:
        model.cuda()
    model.eval()

    start_fol = 10
    end_fol = 12

    if args.target == 'private':
        start_fol = 13
        end_fol = 15
    
    
    for folder_num in range (start_fol,end_fol+1):
        pth = os.path.join(root_dir,str(folder_num), 'audio')
        files = glob(pth + "/*")
        #print("folder_num:{}".format(folder_num))
        for file in sorted(files):
            
            count_pred=[0,0,0,0]
            seqence=file.split(os.path.sep)[-1].split("_")[0]
            #print("file_num:{}".format(seqence), end='')
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
                
                output=model(audio)
                _,pred=torch.max(output,1)
                count_pred[pred.item()]+=1
            if  count_pred[1]>5 or count_pred[2]>5 or count_pred[3]>5:
                final_pred=count_pred[1:4].index(max(count_pred[1:4]))+1
            else:
                final_pred=0
            #print(" pred filling type: {}".format(final_pred))
            datalist=[folder_num,seqence,-1,final_pred,-1]
            with open('./submission.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(datalist)

    elapsed_time = time.time() - start
    print("elapsed_time:{}".format(elapsed_time) + "sec")