
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

from torch.autograd import Variable
from torch.utils.data.dataset import Subset

from model import Net
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

import matplotlib.pyplot as plt
from Mydataset import MyDataset
from myutils import make_dir

import numpy as np

def loss_function(output, target, inv_E, loss_type = 'CE', reduction = 'sum', gamma=2,):
    
    

    loss = None
    eps = 1e-7
    if loss_type == 'CE':
        criterion = nn.CrossEntropyLoss()
        return criterion(output,target)
    
    
    else:
        # focal loss
        target = nn.functional.one_hot(target.to(torch.int64), num_classes=4)
        m = torch.nn.Sigmoid()
        output = m(output)
        output = torch.clamp(output, eps, 1-eps)
        loss = (1-output)**gamma * torch.log(output)

        loss = -1 * target.float() * inv_E.float() * loss
        if reduction == 'sum':
            return torch.sum(loss)
        else:
            return torch.mean(torch.sum(loss, 1))


def log_inverse_class_frequency(n, N):
    return torch.log(N/n+1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training [default: 32]')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate [default: 1e-5]')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging validation result [default: 10]')   
    parser.add_argument('--train_type', action='store', type = str, default='all',
                        choices=['all', 'part'],
                        help='train with all train data or use part of train data as validation [default: all]')
    parser.add_argument('--loss_type', action='store', type = str, default = 'CE',
                        choices=['FL', 'CE'],
                        help='loss type (Focal loss/Cross entropy) [default: CE]')
    parser.add_argument('--reduction', action = 'store', type = str, default='mean',
                        choices=['sum', 'mean'],
                        help='reduction type of loss [default: mean]')
    parser.add_argument('--root', action='store', type = str, default='../data',
                        help = 'root directory of the dataset')
    parser.add_argument('--val', type=int, default=-1, metavar='N', 
                        choices=[-1,1,2,3,4,5,6,7,8,9],
                        help = 'which train folder is used as validation (not trained) [default: -1]')
    args = parser.parse_args()
    
    CLASS_NUM = 4
    CHANNEL_NUM = 8
    
    use_cuda = torch.cuda.is_available()
    root_dir = args.root
    results_dir = os.path.join(root_dir, 'results')
    os.makedirs(results_dir,exist_ok=True)
    
    train_indices = []
    val_indices = []
    mydataset = MyDataset2(root_pth=root_dir, test=False)
    n_samples = len(mydataset)
    folder_count = np.load(os.path.join(root_dir, 'audio', 'folder_count.npy')).tolist()
    
    if args.train_type == 'part' and args.val != -1:
        l = list(range(0, n_samples))
        total_num = 0
        for i, num in enumerate(folder_count):
            if i != args.val-1:
                train_indices += l[total_num:total_num+num]
            else:
                val_indices += l[total_num:total_num+num]
            total_num += num
        print(folder_count)
        print("train_indices",len(train_indices))
        print("val_indices",len(val_indices))
    else:
        # args.train_type == 'all' or args.val != -1
        train_indices = list(range(0, n_samples))  
        val_indices = (np.random.choice(train_indices, 1000, replace=False)).tolist()
    
    train_dataset = Subset(mydataset, train_indices)
    val_dataset =  Subset(mydataset, val_indices)
    
    train_loader   = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=args.batch_size, 
                                                 shuffle=True,
                                                 drop_last = True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.batch_size, 
                                                 shuffle=True)
    
    model     = Net()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,  weight_decay=1e-5)
    if use_cuda:
        model.cuda()
    train_loss_list = []
    val_loss_list   = []
    train_size  = len(train_dataset)
    each_class_size = mydataset.get_each_class_size()
    each_class_size = torch.tensor(each_class_size,dtype=torch.float)
    if use_cuda:
        each_class_size = each_class_size.cuda()
    inv_E = log_inverse_class_frequency(each_class_size, train_size)
    def train(epoch):
        model.train()
        loss_train = 0.0
        correct_train=0
        for batch_idx, (audio, target) in enumerate(train_loader):
            if use_cuda:
                audio = audio.cuda()
                target = target.cuda()
            pdb.set_trace()
            audio = Variable(audio)
            target = Variable(target)
            
            optimizer.zero_grad()
            outputs = model(audio)
            _,preds=torch.max(outputs,1)
            
            loss=loss_function(outputs, target, inv_E=inv_E, loss_type= args.loss_type, reduction=args.reduction)
            loss_train += loss.item()
            correct_train+=torch.sum(preds==target).item()

            loss.backward()
            optimizer.step()
            
        print("Train Epoch {}/{} Loss:{:.4f} Acc:{:.3f}%".format(epoch,args.epochs,loss_train/len(train_loader),correct_train/len(train_loader)/args.batch_size*100))
            
           
    def test(epoch):
        model.eval()
        loss_test = 0
        correct_test=0
        for batch_idx, (audio, target) in enumerate(tqdm(val_loader)):
            if use_cuda:
                audio = audio.cuda()
                target = target.cuda()
            with torch.no_grad():
                audio = Variable(audio)
                target = Variable(target)

                

                outputs = model(audio)
                _,preds=torch.max(outputs,1)
            
                loss = loss_function(outputs, target, inv_E=inv_E, loss_type= args.loss_type, reduction=args.reduction)
                
                loss_test += loss.item()
                correct_test+=torch.sum(preds==target).item()
            
          
        print("Test Epoch {}/{} Loss:{:.4f} Acc{:.3f}%".format(epoch,args.epochs,loss_test/len(val_loader),correct_test/len(val_loader)/args.batch_size*100))
            
    print('use_cuda: ',     use_cuda)
    
    print('batch-size',     args.batch_size)
    print('epochs',         args.epochs)
    print('lr', args.lr)
    print('log-interval',   args.log_interval)
    print('train-type',     args.train_type)
    print('loss-type',      args.loss_type)
    print('reduction',      args.reduction)
    print('val',            args.val)

    print('sample num: ',   n_samples)
    print('length of train dataset: ',  len(train_dataset))
    print('length of val dataset: ',    len(val_dataset))
    print('each_class_size: ',          each_class_size)
    print('log_inverse_class_frequency: ', inv_E)

    
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        if epoch % args.log_interval==0 :
            test(epoch)
            
    save_checkpoint={
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'epochs'    : args.epochs,
            'train_type': args.train_type,
            'loss_type' : args.loss_type,
            'val'       : args.val,
        
        }
    torch.save(save_checkpoint, "{}/{}_{}_{}_{}.pth".format(results_dir,args.train_type,args.val,args.loss_type,args.epochs))
        