import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os



class MyDataset(torch.utils.data.Dataset):
    def __init__(self,root_pth,test=False,transform = None):
        class_num=4
        self.audio_pth = os.path.join(root_pth, 'audio', 'mfcc')
        filling_type = np.load(os.path.join(root_pth, 'audio', 'filling_type.npy'))
        pouring_or_shaking = np.load(os.path.join(root_pth,  'audio', 'pouring_or_shaking.npy'))
        self.label = filling_type * pouring_or_shaking
        self.is_test=test
        self.each_class_size = []
        for i in range(class_num):
            self.each_class_size.append(np.count_nonzero(self.label==i))
        mx=0
        mn=1000
        self.inputs=[]
        for idx in range(self.label.shape[0]):
            data=np.load(os.path.join(self.audio_pth, "{0:06d}".format(idx+1) + '.npy'), allow_pickle=True)
            tmp_max=np.max(data)
            tmp_min=np.min(data)
            if mx<tmp_max:
                mx=tmp_max
            if mn>tmp_min:
                mn=tmp_min
            data= (data-mn)/(mx-mn)
            self.inputs.append(data)
        self.mn=mn
        self.mx=mx
            
    def __len__(self):
        return self.label.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        lbl = -1

        if self.is_test is False:
            lbl = self.label[idx]
            
        output=self.inputs[idx].transpose(2,0,1)
        output=torch.from_numpy(output.astype(np.float32))
        return output , lbl
            
    def get_each_class_size(self):
        return np.array(self.each_class_size)

    
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type = str, default='../data')
    root_pth = args.root
    
    mydataset=MyDataset(root_pth)
    
    
    
    
    
    
    
    
    
    