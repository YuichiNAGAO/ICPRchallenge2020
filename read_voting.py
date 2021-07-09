import json
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', action='store', type = str, default='./data')
    parser.add_argument('--folder-num', action='store', type = int, default=-1)
    parser.add_argument('--file-name', action = 'store', type = str, default = '-1')
    args = parser.parse_args()
    root_dir = args.root
    df = {}
    with open (os.path.join(root_dir, "voting", "voting.json")) as f:
        df = json.load(f)

    result_dir = os.path.join(root_dir, "voting")
    os.makedirs(result_dir, exist_ok=True)

    def save_fig(idx):
        
        # if  df[idx]["sequence"] == df[idx]["final_pred"]:
        #     return
        count_pred = df[idx]["count_pred"]
        pred = df[idx]["pred"]
        # print('frequency:',count_pred)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.tick_params(bottom=False,
                left=True,
                right=False,
                top=False)
        ax.tick_params(labelbottom=False,
                labelleft=True,
                labelright=False,
                labeltop=False)
        
        ax.hist(np.array(pred), bins=range(5), rwidth=0.8)
        ax.set_xlabel('empty                    pasta                   rice                    water')
        ax.set_ylabel('freq')

        plt.savefig(os.path.join(result_dir, "{}_hist.png".format(idx)))
        plt.close()
        # print("./hist.png is saved")
        height = 15
        width = len(pred*2)
        # width = 300 # if you want to fix the width of the graph, please switch to this line
        img = np.zeros((height, width, 3))
        for i in range(len(pred)):
            pix = width/len(pred)
            if pred[i] == 0:
                img[:, int(i * pix):int((i+1)*pix)] = [0,0, 0]#emply:black
            elif pred[i] == 1:
                img[:, int(i * pix):int((i+1)*pix)] = [255,215,0]#pasta:red
            elif pred[i] == 2:
                img[:, int(i * pix):int((i+1)*pix)] = [173,255,47]#rice:green
            elif pred[i] == 3:
                img[:, int(i * pix):int((i+1)*pix)] = [51,0,204]#water:blue
            
        cv2.imwrite(os.path.join(result_dir, "{}_bar.png".format(idx)), img)
        # print("./bar.png is saved")
        print('filename:{}, ground truth:{}, predected:{}, frequency:{}'.format(idx, df[idx]["sequence"], df[idx]["final_pred"], count_pred))



    if args.folder_num == -1:
        for idx in df:
            save_fig(idx)
    elif args.folder_num != -1 and args.file_name != '-1':
        idx = "{}_{}".format(args.folder_num, args.file_name)
        save_fig(idx)

        
    