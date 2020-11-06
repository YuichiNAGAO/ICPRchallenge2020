import cv2
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as T
import pdb
from sklearn.ensemble import IsolationForest



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
        self.cap = cv2.VideoCapture(path)
        self.path=path
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_frame = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) 
        self.video_len_sec = self.video_frame / self.fps
        self.step=step
        self.num_detected=np.zeros(int(self.video_frame//step+1))
        self.size_mask=np.zeros(int(self.video_frame//step+1))
        
        
        
    def processing(self,DT,draw_or_not=False):
        frame_count=0
        while True:
            ret, frame = self.cap.read()
            if ret == True:
                if frame_count%self.step==0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    outputs=DT.predict(frame_rgb)
                    if not human_exists(outputs):
                        self.num_detected[frame_count//self.step]=0
                        frame_count += 1
                        continue
                    roi_list=roi_extract(outputs)
                    self.num_detected[frame_count//self.step]=len(roi_list)
                    if not roi_list:
                        self.size_mask[frame_count//self.step]=0
                    else:
                        self.size_mask[frame_count//self.step]=DT.num_mask_pixel(roi_list[0])
                    if draw_or_not:
                        DT.draw_bb(roi_list,frame)
                        for roi in roi_list:
                            DT.draw_mask(roi)
                frame_count += 1

            else:
                break
    
    
    def get_num_detected(self):
        return self.num_detected
    
    def get_size_mask(self):
        return self.size_mask
    
    def reselect(self,view,DT,frame_num):
        newpath=self.path.replace("c1",view) 
        cap = cv2.VideoCapture(newpath)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            outputs=DT.predict(frame_rgb)
            roi_list=roi_extract(outputs)
            #DT.draw_bb(roi_list,frame)
            #DT.draw_mask(roi_list[0])
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
            
            
def choose_frame(n_detected,n_maskedpixel):
    
    combined=np.concatenate([np.arange(len(n_detected)).reshape(1,-1),n_detected.reshape(1,-1),n_maskedpixel.reshape(1,-1)],axis=0)
    final_order=None
    for total_detected in np.unique(combined[1,:]):
        if total_detected==0:
            continue
        extracted=combined[:,combined[1,:]==total_detected]
        order=extracted[0,:][np.argsort(extracted[2,:])[::-1]]
        if final_order is None:
            final_order=order
        else:
            final_order=np.append(final_order,order)
    
    return final_order


def make_pointcloud(rgb_mask,param,im_depth):
    stack_cord=None
    masked_depth=rgb_mask[1]/255*im_depth
    intrinsic=param[0]["rgb"]
    r_mat=param[1]["rgb"]["rvec"]
    t_vec=param[1]["rgb"]["tvec"]
    for i,row in enumerate(masked_depth):
        for j,pixel in enumerate(row):
            if pixel==0:
                continue
            norm_cord=np.dot(np.linalg.inv(intrinsic),np.array([j,i,1]).reshape(3,1))
            camera_cord=norm_cord *pixel/1000
            world_cord=np.dot(np.linalg.inv(r_mat),camera_cord-t_vec).reshape(1,3).astype(np.float64)
            if stack_cord is None :
                stack_cord=world_cord
            else:
                stack_cord=np.concatenate([stack_cord, world_cord], 0)
    return stack_cord

def outiers_processing(point_data):
    clf = IsolationForest()
    clf.fit(point_data)
    y_pred = clf.predict(point_data)
    point_data_normal=point_data[y_pred==1]

    clf_xy = IsolationForest()
    clf_xy.fit(point_data_normal[:,:2])
    pred_xy = clf_xy.predict(point_data_normal[:,:2])

    clf_xz = IsolationForest()
    clf_xz.fit(point_data_normal[:,[0,2]])
    pred_xz = clf_xz.predict(point_data_normal[:,[0,2]])

    clf_yz = IsolationForest()
    clf_yz.fit(point_data_normal[:,1:])
    pred_yz = clf_yz.predict(point_data_normal[:,1:])

    pred_xyz=np.logical_and(pred_xy==1,pred_xz==1,pred_yz==1)

    point_data_normal=point_data_normal[pred_xyz]
    
    return point_data_normal


def volume_by_world2image(point_data_normal,param,mask):
    x_max,y_max,z_max=np.max(point_data_normal,axis=0)
    x_min,y_min,z_min=np.min(point_data_normal,axis=0)
    xx,yy,zz = np.meshgrid(np.arange(x_min,x_max,0.01), np.arange(y_min,y_max,0.01),np.arange(z_min,z_max,0.01))
    xyz_grid = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1),zz.reshape(-1, 1)], axis=1)
    volume_box=(x_max-x_min)*(y_max-y_min)*(z_max-z_min)*1000000
    homo_world_coord=np.concatenate([xyz_grid,np.ones((xyz_grid.shape[0],1))],axis=1)
    homo_world_coord=np.transpose(homo_world_coord)
    proj_mat=param[1]["rgb"]['projMatrix']
    homo_image_coord=np.dot(proj_mat,homo_world_coord)
    homo_image_coord=np.transpose(homo_image_coord)
    uv1=homo_image_coord/homo_image_coord[:,2].reshape(-1,1)
    uv=uv1[:,:2]
    
    uv_round=np.round(uv).astype(np.int64)
    
    correct_v,correct_u=np.where(mask!=0)
    uv_pair=np.concatenate([correct_u.reshape(-1,1),correct_v.reshape(-1,1)],axis=1)
    count=0
    for uv_pred in uv_round:
        if np.any(np.all(uv_pred==uv_pair,axis=1)):
            count+=1 
    return volume_box*(count/len(uv_round))
    
            

