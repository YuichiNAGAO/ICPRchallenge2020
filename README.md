# ICPRchallenge2020
A solution for ICPR 2020 CORSMAL Challenge

## Environment
cuda 10.2

To innstall libraries using anaconda, please execute this:

```
$ bash conda_install.sh
```

## How to execute scripts

### 1. Preparation
Please confirm that `annotations.csv` is placed under the current directory.
Please create a new directory that you can put datasets and put them uder that directory. The simplest way is create a directory called `data` by typing `mkdir data` on command line.

### 2. Data pre-processing
```
$ audio_processing.py --root [path to the dataset]
```
The preprocessed data is stored under the data directory.

### 3. Training
```
$ train.py --root [path to the dataset]
```
other options:
```
parser.add_argument('--root', action='store', type = str, default='./data',
                    help = 'root directory of the dataset')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training [default: 32]')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                    help='learning rate [default: 1e-5]')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging validation result [default: 10]')   
parser.add_argument('--train_type', action='store', type = str, default='all',
                    choices=['all', 'part'],
                    help='train with all train data or use part of train data as validation [default: all]')
parser.add_argument('--val', type=int, default=-1, metavar='N', 
                    choices=[-1,1,2,3,4,5,6,7,8,9],
                    help = 'which train folder is used as validation (not trained) [default: -1]')
parser.add_argument('--loss_type', action='store', type = str, default = 'CE',
                    choices=['FL', 'CE'],
                    help='loss type (Focal loss/Cross entropy) [default: CE]')
parser.add_argument('--reduction', action = 'store', type = str, default='mean',
                    choices=['sum', 'mean'],
                    help='reduction type of loss [default: mean]')
```
If you choose `part` for the train type, please make sure to choose a validation folder(i.e., container). Then the chosen folder is removed for training.
If you choose `all`, the the model is trained using all of the train data.
Basically, all of the default options are optimized, so you just need to write path to data directory. For example,
```
$ train.py --root './data'
```

At the end of training, weights of trained model are saved under `[data dorectory]/results`.


### 4. Testing
```
$ test.py --root [path to the dataset]
```
other options:
```
parser.add_argument('--root', action='store', type = str, default='./data')
parser.add_argument('--folder_num', action='store',type=int, nargs='*',default=[10, 11, 12],help='name of folder that you want to test')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='please set the same number as you have put in training phase!!')
parser.add_argument('--train_type', action='store', type = str, default='all',
                    choices=['all', 'part'],
                    help='please set the same string as you have put in training phase!!')
parser.add_argument('--val', type=int, default=-1, metavar='N', 
                    choices=[-1,1,2,3,4,5,6,7,8,9],
                    help ='please set the same number as you have put in training phase!!')
parser.add_argument('--loss_type', action='store', type = str, default = 'CE',
                    choices=['FL', 'CE'],
                    help='please set the same string as you have put in training phase!!')   
```
You have to specify folder number (container number) for which you want to make the prediction.
Also, as for "epochs", "train_type", "val", "loss_type", you need to use the same settings that you have set in train.py.
Basically, of the default options are optimized (the same default as train.py).
Example:
```
$ train.py --root './data' --folder_num 13 14 15
```

At the end of testing, `submission.csv` containing the predictions is created under the current directory.

