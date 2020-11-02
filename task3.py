
import glob
import argparse
import cv2
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from PIL import Image


import torch
import torchvision
import torchvision.transforms as T

import pdb

class Detection:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detectionModel = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.detectionModel.eval()
        self.detectionModel.to(self.device)
        self.trf = T.Compose([T.ToTensor()])
        self.coco_names = ['unlabeled', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


    def predict(self,im):
        return self.detectionModel([self.trf(im).to(self.device)])[0]
        
    def draw_bb(self,roi_list,image):
        for roi in roi_list:
            box=roi[2]
            c1, c2 = (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item()))
            label=roi[0]
            score=roi[1]
            display_txt = "%s: %.1f%%" % (self.coco_names[label], 100*score)
            tl=3
            color=(0,0,255)
            cv2.rectangle(image,c1,c2 ,color,thickness=tl)
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(display_txt, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c2, color, -1)  # filled
            cv2.putText(image, display_txt, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        bgr_im_show(image)
        
    def draw_mask(self,roi):
        print(self.coco_names[roi[0]])
        plt.imshow(roi[3],cmap='Greys')
        plt.show()
    
    def num_mask_pixel(self,roi):
        return np.sum(roi[3]==255)
    
class Video_processing:
    def __init__(self,path,step):
        cap = cv2.VideoCapture(path)
        self.path=path
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.video_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT) 
        self.video_len_sec = self.video_frame / self.fps
        self.step=step
        self.num_detected=np.empty((3,int(self.video_frame//step+1)))
        
        
    def processing(self,view,DT):
        newpath=self.path.replace("c1",view) 
        row=int(view[-1])-1
        cap = cv2.VideoCapture(newpath)
        frame_count=0
        while True:
            ret, frame = cap.read()
            if ret == True:
                if frame_count%self.step==0:
                    print("frame_count{}".format(frame_count))
                    print("{}s".format(frame_count/self.fps))
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    outputs=DT.predict(frame_rgb)
                    if not human_exists(outputs):
                        print("Human doesn't exist")
                        self.num_detected[row][frame_count//self.step]=0
                        frame_count += 1
                        continue
                    roi_list=roi_extract(outputs)
                    self.num_detected[row][frame_count//self.step]=len(roi_list)
                    DT.draw_bb(roi_list,frame)
                    for roi in roi_list:
                        DT.draw_mask(roi)
                frame_count += 1

            else:
                break
    
    
    def get_num_detected(self):
        return self.num_detected
    
    def reselect(self,view,DT,frame_num):
        newpath=self.path.replace("c1",view) 
        cap = cv2.VideoCapture(newpath)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            outputs=DT.predict(frame_rgb)
            roi_list=roi_extract(outputs)
        return  [frame_rgb,roi_list[0][3]]


def roi_extract(output):
    roi_list=[]
    width=1280
    #high_score_container=True
    for i in range(0, len(output['labels'])):
        if  output['boxes'][i][2]<=width/4 or output['boxes'][i][0]>=width/4*3:
            continue
        if  output['labels'][i] in [44, 46, 47, 86]:
            #if output['scores'][i]>0.9:
             #   high_score_container=True
            seg = np.uint8(255.*output['masks'][i,:,:,:].detach().cpu().numpy().squeeze())
            seg = ((seg >= 128)*255).astype(np.uint8)
            roi_list.append([output['labels'][i].item(),output['scores'][i].item(),output['boxes'][i],seg])
    return roi_list


def human_exists(output):
    width=1280
    for i in range(0, len(output['labels'])):
        if  output['labels'][i]==1:
            center_x=(output['boxes'][i][0]+output['boxes'][i][2])/2
            if center_x>=width/3 and center_x<=width/3*2:
                return True
            else:
                return False 

def bgr_im_show(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show() 
            
            
def choose_frame(n_detected):
    nul_frame=[]
    for col in range(n_detected.shape[1]):
        if np.any(n_detected[:,col]==0):
            n_detected[:,col]=None
            
    summed=np.sum(n_detected, axis=0)
    return np.argsort(summed)

def make_pointcloud(rgb_mask,param,im_depth):
    stack_cord=None
    img_rgb=rgb_mask[0]
    masked_depth=rgb_mask[1]/255*im_depth
    intrinsic=param[0]["rgb"]
    r_mat=param[1]["rgb"]["rvec"]
    t_vec=param[1]["rgb"]["tvec"]
    for i,row in enumerate(masked_depth):
        for j,pixel in enumerate(row):
            if pixel==0:
                continue
            norm_cord=np.dot(np.linalg.inv(intrinsic),np.array([j,i,3]).reshape(3,1))
            camera_cord=norm_cord *pixel
            world_cord=np.dot(np.linalg.inv(r_mat),camera_cord-t_vec).reshape(1,3)
            world_cord=np.concatenate([world_cord,img_rgb[i][j].reshape(1,3)],1).astype(np.float32)
            if stack_cord is None :
                stack_cord=world_cord
            else:
                stack_cord=np.concatenate([stack_cord, world_cord], 0)
    return stack_cord
    





        
if __name__="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type = str, default='./data')
    parser.add_argument('--step', type = int, default=12)
    args = parser.parse_args()
    
    root_dir = args.root
    step=args.step
    
    DT=Detection()  
    os.makedirs('{}/pointdata'.format(root_dir), exist_ok=True)
    
    for container in range(1,10):
        os.makedirs('{}/pointdata/{}'.format(root_dir,container), exist_ok=True)
        folder="{}/{}".format(root_dir,container)
        for path in sorted(glob.glob(folder+"/rgb/*")):
            filename=os.path.splitext(os.path.basename(path))[0]
            if not filename.split("_")[-1]=="c1":
                continue
            print(path)
            VP=Video_processing(path,step)
            for view in ["c1","c2","c3"]:
                VP.processing(view,DT)
            n_detected=VP.get_num_detected()
            which_frame=choose_frame(n_detected)*step
            best_frame=which_frame[0]
            image_store=[]
            param_store=[]
            depth_store=[]
            for view in ["c1","c2","c3"]:
                image_store.append(VP.reselect(view,DT,best_frame))

                calib_path=os.path.splitext(path)[0].replace("rgb","calib").replace("c1",view)+"_calib"+'.pickle'
                with open(calib_path, 'rb') as f:
                    u = pickle._Unpickler(f)
                    u.encoding = 'latin1'
                    intrinsic,extrinsic,_,_ = u.load()
                param_store.append([intrinsic,extrinsic])

                depth_path=folder+"/depth/"+filename.rsplit("_",1)[0]+"/{}/{}.png".format(view,str(best_frame).zfill(4))
                depth_store.append(cv2.imread(depth_path,-1))

            point_data=None
            for rgb_mask,param,depth_img in zip(image_store,param_store,depth_store):
                if point_data is None:
                    point_data=make_pointcloud(rgb_mask,param,depth_img)
                    #point_data=np.array([["X","Y","Z","R","G","B"]],dtype=str)
                    #point_data=np.concatenate([point_data, make_pointcloud(rgb_mask,param,depth_img)], 0)
                else:
                    point_data=np.concatenate([point_data, make_pointcloud(rgb_mask,param,depth_img)], 0)

            np.savetxt('{}/pointdata/{}/{}.csv'.format(root_dir,container,filename.rsplit("_",1)[0]), point_data)
            break
        break
    