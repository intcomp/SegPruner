import gc
import os
import warnings

import math

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
import json
import glob
import torch
from PIL import Image
import copy
import numpy as np
from tqdm import tqdm
import argparse

from llava.mm_utils import (
    get_model_name_from_path,
    tokenizer_image_token,
    process_images,
    KeywordsStoppingCriteria
)
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

from llava.model.builder import load_pretrained_model


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--visual_token_num", type=int, default=393, help="visual_token_num.")
    parser.add_argument("--image_aspect_ratio", type=str, default=None, help="Aspect ratio setting.")
    parser.add_argument("--r", type=float, default=0, help="Token merging ratio.")
    parser.add_argument("--frames_root", default="/data/ScanQA/frames12_square", help="images path")
    parser.add_argument("--json_path", default="/data/ScanQA/qa/ScanQA_v1.0_val.json", help="jsons path")
    parser.add_argument("--lam", type=float, default=0.5, help="lam")
    args = parser.parse_args()
    return args

def build_inputs_for_multimage(question, pil_images):
    question = f"{DEFAULT_IMAGE_TOKEN}" * len(pil_images) + f"{question} \n\n"
    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], question) 
    conv.append_message(conv.roles[1], None) 
    prompt = conv.get_prompt()  

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)

    image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in pil_images]

    return input_ids, image_tensors, conv

def resize_and_pad_image(image, target_resolution):
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    # Determine which dimension (width or height) to fill
    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        # Width will be filled completely
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        # Height will be filled completely
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image 
    resized_image = image.resize((new_width, new_height), resample=Image.NEAREST)

    # Create a new image with the target size and paste the resized image onto it
    if image.mode=="RGB":
        fill_color = (0, 0, 0)
    else:
        fill_color = 0

    new_image = Image.new(image.mode, (target_width, target_height), fill_color)
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def load_image(file):
    image = Image.open(file)
    # preprocess
    image = resize_and_pad_image(image, [384, 384])
    image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

    return image

def load_pose(filename):
    lines = open(filename).read().splitlines()
    assert len(lines) == 4
    lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]
    lines = np.asarray(lines).astype(np.float32)
    lines = torch.from_numpy(lines).float().to(device)
    return lines

def load_depth(file):
    depth_image = Image.open(file)
    # preprocess
    depth_image = resize_and_pad_image(depth_image, [384,384])
    depth_image = np.array(depth_image).astype(np.float32) / 1000.0
    depth_image = torch.from_numpy(depth_image).float().to(device)
    return depth_image

def list_first_n_frames(scene_dir):
    all_imgs = sorted(glob.glob(os.path.join(scene_dir, "*")))
    return all_imgs


@torch.inference_mode()
def answer_one_sample(question_text, image_paths):
    new_image_paths = list_first_n_frames(os.path.join(image_paths, "color"))
    depth_paths = list_first_n_frames(os.path.join(image_paths, "depth"))
    pose_paths = list_first_n_frames(os.path.join(image_paths, "pose"))

    imgs = [load_image(p) for p in new_image_paths]
    image_sizes = [Image.open(img).size for img in new_image_paths]
    depths = [load_depth(d) for d in depth_paths]
    poses = [load_pose(po) for po in pose_paths]

    intrinsic = [[346.5546, 0, 191.5, 0], [0, 347.238, 143.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    input_ids, images,conv = build_inputs_for_multimage(question_text, imgs)

    output_ids, visual_token_num = model.generate(
        inputs=input_ids,
        images=images,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=128,
        depths=depths,
        poses=poses,
        intrinsic=intrinsic
    )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    return outputs[0], visual_token_num


def main():
    assert os.path.exists(json_path), f"JSON not found: {json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    results = []
    total = len(annotations)
    print(f"Total samples to run: {total}")

    pbar = tqdm(range(total), desc="Evaluating", ncols=100, dynamic_ncols=True)
    correct = 0
    incorrect = 0
    for idx, ann in zip(pbar, annotations):
        qid = ann.get("question_id")
        scene_id = ann.get("scene_id")
        question = ann.get("question", "").strip()
        gt_answers = ann.get("answers", "")
        if isinstance(gt_answers, str):
            gt_list = [gt_answers] if gt_answers else []
        elif isinstance(gt_answers, list):
            gt_list = gt_answers
        else:
            gt_list = []

        image_paths = os.path.join(frames_root, scene_id)

        ans, visual_token_num_ans= answer_one_sample(question, image_paths)

        # correct rate
        if any(ans.lower() == gt.lower() for gt in gt_list):
            correct += 1
        else:
            incorrect += 1
        acc = correct / (correct + incorrect)
        img_paths = list_first_n_frames(os.path.join(frames_root, scene_id, "color"))
        results.append({
            "question_id": qid,
            "scene_id": scene_id,
            "question": question,
            "answer": ans,
            "frames_used": [os.path.basename(p) for p in img_paths],
            "EM@1":acc,
            "visual_token_num_ans":visual_token_num_ans,
            "visual_token_num":visual_token_num,
            "important_ratio":important_ratio
        })
        pbar.set_postfix(acc=f"{acc:.4f}")


    print(f"accuracy:{acc:.4f}")

    # save results
    out_path = os.path.join("/data/ScanQA", "answers")

    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(out_path,f"visual_token_num:{visual_token_num}_r:{important_ratio:.1f}_lam:{lam:.1f}_acc:{acc:.4f}.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    args = parse_args()
    model_path = "/data/model/llava-onevision-qwen2-7b-ov"
    model_name = get_model_name_from_path(model_path)

    frames_root = args.frames_root
    json_path = args.json_path
    lam = args.lam
    patch_size = 14
    num_patches = 384 // patch_size

    NUM_FRAMES_PER_SCENE = 12

    visual_token_num = args.visual_token_num
    important_ratio = args.r

    device = "cuda"
    device_map = "auto"
    llava_model_args = {
        "multimodal": True,
    }
    overwrite_config = {}
    overwrite_config["image_aspect_ratio"] = args.image_aspect_ratio
    llava_model_args["overwrite_config"] = overwrite_config

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        None,
        model_name=model_name,
        device_map="auto",
        visual_token_num=visual_token_num,
        important_ratio=important_ratio,
        lam=lam,
        **llava_model_args
    )
    model = model.eval()

    main()
    del model  
    gc.collect()
    torch.cuda.empty_cache()
