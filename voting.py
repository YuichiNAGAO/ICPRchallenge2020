from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from model import Net, LSTMNet
from AudioProcessing import AudioProcessing
from Mydataset import Padding
from utils_t3 import *

import numpy as np
import scipy.io.wavfile
import os
from tqdm import tqdm
import pandas as pd
from glob import glob
import csv
import pdb
import pickle

import torch
import time
import random
import json

"""
$ python test.py --root D:\codes\dataset --folder_num 13 14 15
"""

def load_checkpoint_T2(file_path, use_cuda=False):
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)
    model = Net()
    model.load_state_dict(checkpoint['state_dict'])
    return model

def load_checkpoint_T1(file_path, use_cuda=False):
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)
    model = LSTMNet(is_cuda=use_cuda,
                    input_dim=checkpoint['input_dim'],
                    output_dim=checkpoint['class_num'],
                    hidden_dim=checkpoint['hidden_dim'],
                    n_layers=checkpoint['n_layers'],
                    drop_prob=checkpoint['drop_prob'])
    model.load_state_dict(checkpoint['state_dict'])
    return model


"""
Constant 
"""
mfcc_MAX_VALUE=194.19187653405487
mfcc_MIN_VALUE=-313.07119549054045

t2_MAX_VALUE = 57.464638
t2_MIN_VALUE = -1.1948369
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
    parser.add_argument('--t2_epochs', type=int, default=200, metavar='N',
                        help='please set the same number as you have put in training phase!! number of epochs to train [default:200]')
    args = parser.parse_args()
    
    start = time.time()
    root_dir = args.root
    voting_dir = os.path.join(root_dir, 'voting')
    os.makedirs(voting_dir, exist_ok=True)
    model_dir = os.path.join(root_dir, './models')
    
    
    use_cuda = torch.cuda.is_available()
    model_T2=load_checkpoint_T2(os.path.join(model_dir, "T2_{}_{}_{}_{}.pth".format(args.train_type,args.val,args.loss_type,args.t2_epochs)), use_cuda=use_cuda)


    if use_cuda:
        model_T2.cuda()
    model_T2.eval()
    DT=Detection() 
    save_data = {}
    data_num = 0
    for folder_num in args.folder_num:
        pth = os.path.join(root_dir,str(folder_num), 'audio')
        pth_rgb=os.path.join(root_dir,str(folder_num), 'rgb')
        files = glob(pth + "/*")
        #print("folder_num:{}".format(folder_num))

        for file in sorted(files):
            # print(file)
            predlist = []
            datalist = []
            count_pred=[0,0,0,0]
            print(file)
            if folder_num >= 10:
                sequence = -1

            else:
                sequence=int(file.split(os.path.sep)[-1].split("_")[1][2])
            #sequence=-1
            #print("file_num:{}".format(sequence), end='')
            sample_rate, signal = scipy.io.wavfile.read(file)
            ap = AudioProcessing(sample_rate,signal)
            mfcc = ap.calc_MFCC()
            mfcc_length=mfcc.shape[0]
            f_step=int(mfcc.shape[1]*0.25)
            f_length=mfcc.shape[1]
            save_mfcc_num=int(np.ceil(float(np.abs(mfcc_length - save_size)) /f_step))
            for i in range(save_mfcc_num):
                tmp_mfcc = mfcc[i*f_step:save_size+i*f_step,: ,:]
                tmp_mfcc= (tmp_mfcc-mfcc_MIN_VALUE)/(mfcc_MAX_VALUE-mfcc_MIN_VALUE)
                tmp_mfcc=tmp_mfcc.transpose(2,0,1)
                audio=torch.from_numpy(tmp_mfcc.astype(np.float32))
                audio=torch.unsqueeze(audio, 0)
                if use_cuda:
                    audio = audio.cuda()
                #Task 2     
                with torch.no_grad():
                    mid, pred_T2=model_T2.before_lstm(audio)
                    _,pred_T2=torch.max(pred_T2,1)
                    count_pred[pred_T2.item()]+=1
                    predlist.append(pred_T2.item())

                    datalist.append(mid.to('cpu').detach().numpy().copy())
            # print(predlist)
            if  count_pred[1]>5 or count_pred[2]>5 or count_pred[3]>5:
                final_pred_T2=count_pred[1:4].index(max(count_pred[1:4]))+1
            else:
                final_pred_T2=0
            # print(" pred filling type: {}".format(final_pred_T2))
            print("sequence:{}, frequency:{}".format(sequence, count_pred))
            file_name = file.split(os.path.sep)[-1].replace('_audio.wav', '')
            to_save_data = {"data_num":data_num,
                            "folder":folder_num,
                            "file":file_name,
                            "sequence":sequence, 
                            "count_pred":count_pred,
                            "final_pred":final_pred_T2,
                            'pred':predlist}
            save_data["{}_{}".format(folder_num, file_name)] = to_save_data
            data_num+=1
    with open (os.path.join(voting_dir, "voting.json"), 'w') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)        
    elapsed_time = time.time() - start
    print("elapsed_time:{}".format(elapsed_time) + "sec")
