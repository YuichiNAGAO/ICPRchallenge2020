# ICPRchallenge2020
A solution for ICPR 2020 CORSMAL Challenge Task2, filling type classification
URL:http://corsmal.eecs.qmul.ac.uk/ICPR2020challenge.html

## Environment
cuda 10.2

To innstall libraries using anaconda, please execute this:

```
$ bash conda_install.sh
```

The contents of `conda_install.sh`is :
```
conda install pytorch torchvision -c pytorch
conda install -c anaconda numpy
conda install -c anaconda scipy
conda install -c anaconda pillow
conda install -c anaconda cython
conda install -c conda-forge matplotlib
conda install -c anaconda scikit-image
conda install -c conda-forge opencv
conda install -c anaconda ipython
conda install -c conda-forge tqdm
conda install -c anaconda scikit-learn
conda install -c anaconda pandas
```

## How to execute scripts
### For organizers
##### 1. clone this repository using `$ git clone git@github.com:YuichiNAGAO/ICPRchallenge2020.git`
##### 2. Download the pre-trained models from here[]<- TODO: rewrite
##### 3. Create `models` folder in the same directory as the scripts, and place the pre-trained models in the new folder
##### 4. Install the libraries
##### 5. Execute : `$ python test.py --root [path to the dataset]　--folder_num [folder numbers]`
Then, the `submission.csv` containing the predictions is created under the current directory.

#### Expected file structures

```
root of scripts
|----- models
|         |----- T1_all_-1_CE_400.pth(model for task 1)
|         |----- T2_all_-1_CE_200.pth(model for task 2)
|----- train_T1.py
|----- train_T2.py
|----- test.py
|----- model.py
|----- Mydataset.py
|----- annotations.csv
...

root of dataset
|----- 13
|          |----- audio
|          |----- calib
|          |----- depth
|          |----- imu
|          |----- ir
|          |----- rgb
|----- 14
....
```

## Start from the begginig
### 1. Preparation
Please confirm that `annotations.csv` is placed under the current directory.
Please create a new directory that you can put datasets and put them uder that directory. The simplest way is create a directory called `data` by typing `mkdir data` on command line.

### 2. Data pre-processing for task 2
```
$ python AudioProcessing.py --root [path to the dataset]
```
The preprocessed data is stored under `[path to the dataset]/audio`.

### 3. Training for task 2
```
$ python train_T2.py --root [path to the dataset]
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
$ train_T2.py --root './data'
```

At the end of training, weights of trained model are saved under `[path to the dataset]/T1_results`.

### 4. Data pre-processing for task 1 using the result of task 2
```
$ python preprocessing_T1.py --root [path to the dataset]
```
The preprocessed data is stored under `[path to the dataset]/T2_mid`.

### 5. Training for task 1
```
$ python train_T1.py --root [path to the dataset]
```
other options:
```
    parser.add_argument('--root', action='store', type = str, default='./data',
                        help = 'root directory of the dataset')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input batch size for training [default: 16]')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
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
    parser.add_argument('--val', type=int, default=-1, metavar='N', 
                        choices=[-1,1,2,3,4,5,6,7,8,9],
                        help = 'which train folder is used as validation (not trained) [default: -1]')
```

```
$ train_T1.py --root './data'
```

At the end of training, weights of trained model are saved under `[path to the dataset]/T1_results`.

### 4. Testing
```
$ python test.py --root [path to the dataset]　--folder_num 10 11 12
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
Basically, all the default options are optimized (the same default as train.py).
Example:
```
$ train.py --root './data' --folder_num 13 14 15
```

At the end of testing, `submission.csv` containing the predictions is created under the current directory.
