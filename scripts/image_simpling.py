import glob
import shutil

import torch
from PIL import Image
from pathlib import Path
import os

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, get_model_name_from_path

ROOT = "/data/ScanQA/scannetv2/frames_square"
OUT = os.path.join("/data/ScanQA/frames12_square")
txt_path = "/data/ScanQA/qa/scannetv2_val.txt"
scene_list = sorted([line.rstrip() for line in open(os.path.join(txt_path))])

k = 12
dir = "pose"

os.makedirs(OUT, exist_ok=True)

def main():

    scenes = sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))])
    for scene in scene_list:
        color_dir = os.path.join(ROOT, scene)
        if not os.path.isdir(color_dir):
            continue

        imgs = sorted(glob.glob(os.path.join(color_dir,"*")),key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        n = len(imgs)
        if n == 0:
            print(f"{scene}no jpg skip")
            continue

        if n <= k:
            sel = imgs
        else:
            sel = [imgs[round(i*(n-1)/k)] for i in range(k)]

        out_frames = os.path.join(OUT, scene, dir)
        os.makedirs(out_frames, exist_ok=True)
        # save
        for src in sel:
            dst = os.path.join(out_frames, os.path.basename(src))
            shutil.copy2(src, dst)

if __name__ == "__main__":
    main()
