from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import scipy.io.wavfile
import time

"""
$ python preprocessing_T2.py --root D:\codes\dataset 
"""

class AudioProcessing():
    def __init__(self,sample_rate,signal,frame_length_t=0.025,frame_stride_t=0.01,nfilt =64):
        
        self.sample_rate=sample_rate
        self.signal = signal
        self.frame_length_t=frame_length_t
        self.frame_stride_t=frame_stride_t
        self.signal_length_t=float(signal.shape[0]/sample_rate)
        self.frame_length=int(round(frame_length_t * sample_rate)) #number of samples
        self.frame_step=int(round(frame_stride_t * sample_rate))
        self.signal_length = signal.shape[0]
        self.nfilt=nfilt
        self.num_frames = int(np.ceil(float(np.abs(self.signal_length - self.frame_length)) / self.frame_step))
        self.pad_signal_length=self.num_frames * self.frame_step + self.frame_length
        self.NFFT=512
        
    def calc_MFCC(self):
        pre_emphasis=0.97
        emphasized_signal=np.concatenate([self.signal[0,:].reshape([1,-1]),  self.signal[1:,:] - pre_emphasis * self.signal[:-1,:]], 0)
        z = np.zeros([self.pad_signal_length - self.signal_length,8])
        pad_signal = np.concatenate([emphasized_signal, z], 0)
        indices = np.tile(np.arange(0, self.frame_length), (self.num_frames, 1)) + np.tile(np.arange(0, self.num_frames * self.frame_step, self.frame_step), (self.frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        frames=frames*np.hamming(self.frame_length).reshape(1,-1,1)
        frames=frames.transpose(0,2,1)
        mag_frames = np.absolute(np.fft.rfft(frames,self.NFFT))
        pow_frames = ((1.0 / self.NFFT) * ((mag_frames) ** 2))
        filter_banks = np.dot(pow_frames, self.cal_fbank().T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)  # dB
        filter_banks =filter_banks.transpose(0,2,1)
        
        return filter_banks
           
    def cal_fbank(self):
        
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (self.sample_rate / 2) / 700))  
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.nfilt + 2)  
        hz_points = (700 * (10**(mel_points / 2595) - 1)) 
        bin = np.floor((self.NFFT + 1) * hz_points / self.sample_rate)
        fbank = np.zeros((self.nfilt, int(np.floor(self.NFFT / 2 + 1))))
        for m in range(1, self.nfilt + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        return fbank
        
        
        
          
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type = str, default='./data')
    parser.add_argument('--ratio_step', type =float, default=0.25)
    args = parser.parse_args()
    
    start = time.time()


    root_pth = args.root
    df = pd.read_csv('annotations.csv', header = 0)
    df_len=len(df)
    os.makedirs(os.path.join(root_pth, 'audio'), exist_ok=True)
    mfcc_path = (os.path.join(root_pth, 'audio', 'mfcc'))
    os.makedirs(mfcc_path,exist_ok=True)
    count = 0
    pouring_or_shaking_list = []
    file_idx_list = []
    filling_type_list = []
    folder_count = [0]*9
    folder_count_detail = [[] for _ in range(9)]
    pbar = tqdm(total=df_len)
    save_size=64
    
    for fileidx in range(df_len):
        file_name = df.iat[fileidx, 2]
        folder_num = df.iat[fileidx, 0]
        start_time =  df.iat[fileidx, 9]
        end_time = df.iat[fileidx, 10]
        filling_type = df.iat[fileidx, 4]
        
        audio_filename = file_name.rsplit("_", 1)[0] + '_audio.wav'
        audio_path = os.path.join(root_pth, str(folder_num), 'audio', audio_filename)
        sample_rate, signal = scipy.io.wavfile.read(audio_path)
        
        ap = AudioProcessing(sample_rate,signal,nfilt=save_size)
        mfcc = ap.calc_MFCC()
        mfcc_length=mfcc.shape[0]
        if mfcc_length < save_size:
            print("file {} is too short".format(fileidx))
        else:
            f_step=int(mfcc.shape[1]*args.ratio_step)
            f_length=mfcc.shape[1]
            save_mfcc_num=int(np.ceil(float(np.abs(mfcc_length - save_size)) / f_step))
            folder_count_detail[folder_num-1].append(save_mfcc_num)
            for i in range(save_mfcc_num):
                count += 1
                tmp_mfcc = mfcc[i*f_step:save_size+i*f_step,: ,:]
                if start_time == -1:
                    pouring_or_shaking_list.append(0)
                elif start_time/ap.signal_length_t*mfcc_length<i*f_step+f_length*0.75 and end_time/ap.signal_length_t*mfcc_length>i*f_step+f_length*0.25:
                    pouring_or_shaking_list.append(1) 
                else:
                    pouring_or_shaking_list.append(0)
                
                filling_type_list.append(filling_type)
                file_idx_list.append(fileidx)
                
                folder_count[folder_num-1] += 1
                
                np.save(os.path.join(mfcc_path, "{0:06d}".format(count)), tmp_mfcc)
                
        pbar.update()
    np.save(os.path.join(root_pth, 'audio', 'pouring_or_shaking'), np.array(pouring_or_shaking_list) )
    np.save(os.path.join(root_pth, 'audio', 'filling_type'), np.array(filling_type_list))
    np.save(os.path.join(root_pth, 'audio', 'folder_count'), np.array(folder_count))
    np.save(os.path.join(root_pth, 'audio', 'folder_count_detail'), np.array(folder_count_detail))

    elapsed_time = time.time() - start
    print("elapsed_time:{}".format(elapsed_time) + "sec")
            
            
            