import sys
sys.path.append(".")
sys.path.append("..")

import faceAlignment.face_alignment.api as face_alignment
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from os.path import join
import numpy as np
from tqdm import tqdm
import json
from glob import glob
import argparse

import matplotlib.pyplot as plt


class Gen2DLandmarks(object):
    def __init__(self) -> None:
        super().__init__()
        self.fa_func = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        
        
    def main_process(self, img_dir):
        
        img_path_list = [x for x in glob("%s/*.png" % img_dir) if "mask" not in x]
        
        if len(img_path_list) == 0:
            print("Dir: %s does include any .png images." % img_dir)
            exit(0)
        
        img_path_list.sort()

        for img_path in tqdm(img_path_list, desc="Generate facial landmarks"):
            
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            # 这里用cv2将图片读出来转成RGB直接输入到这个api中就行了
            res, landmarks_scores, faces = self.fa_func.get_landmarks(img_rgb, return_bboxes=True)
            
            if res is None:
                print("Warning: can't predict the landmark info of %s" % img_path)
                
            # base_name = img_path[img_path.rfind("/") + 1:-4]
            save_path = img_path[:-4] + "_lm2d.txt"
            preds = res[0]

            # 这段代码我本来是想可视化的, 但是这里输出的时候是进行过特殊的处理的, 将关键点分向了4个角上, 也许是为了后续的处理需要
            # face = faces[0]
            # h, w, c = face.shape
            # img = np.zeros((h, w, c), dtype=np.uint8)
            # for index, p in enumerate(preds):
            #     # cv2.circle(img, i.astype('uint8') - np.array([face_bbox[0], face_bbox[1]]), 1, (255, 255, 255), -1)
            #     cv2.circle(face, p.astype('uint8'), 2, (255, 0, 0), -1)
            #     cv2.circle(img, p.astype('uint8'), 2, (255, 255, 255), -1)
            #     # plt.imshow(np.concatenate([img, face], axis=1))
            # plt.imshow(np.concatenate([face, img], axis=1))
            # plt.show()
            with open(save_path, "w") as f:
                for tt in preds:
                    # [x1, y1, x2, y2,...., xn, yn]
                    f.write("%f \n"%(tt[0]))
                    f.write("%f \n"%(tt[1]))

    def main_process_image(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # 这里用cv2将图片读出来转成RGB直接输入到这个api中就行了
        res = self.fa_func.get_landmarks(img_rgb, return_bboxes=False)
        preds = res[0]
        return preds
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='The code for generating facial landmarks.')
    # parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--img_dir", type=str, required=True)
    args = parser.parse_args()

    tt = Gen2DLandmarks()
    tt.main_process(args.img_dir)
    