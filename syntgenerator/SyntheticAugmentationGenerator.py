import os
import sys
import json
import random

import torch
import argparse
import numpy as np
from PIL import Image, ImageDraw
from tqdm.autonotebook import tqdm
from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting

class AugmentationGenerator():
    def __init__(self, source_json: str or os.PathLike, final_json: str or os.PathLike, dir_images: str or os.PathLike, dir_dataset: str or os.PathLike, weights: str = "stabilityai/stable-diffusion-2-inpainting"):
        self.dir_images = dir_images
        self.img_files = sorted(os.listdir(dir_images))
        with open(source_json) as f:
            annotations = json.load(f)
            self.bboxs = annotations['bboxs']
            self.classes = annotations['classes']

        self.coco = {
            "info":{},
            "images": [],
            "categories": [],
            "annotations": []
        }
        for cls in self.classes:
            category_id = cls['id']
            label = cls['label']
            self.coco["categories"] += [{'supercategory': label, "id": category_id, "name": label}]

        self.final_json = final_json
        self.dir_dataset = dir_dataset
        os.makedirs(self.dir_dataset, exist_ok=True)
        
        self.weights = weights
        try:
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.weights,
                torch_dtype=torch.float16,
                force_download=True, 
                resume_download=False,
                safety_checker = None
            ).to("cuda")
        except ValueError:
            print('The weights of your model are not suitable for the StableDiffusionInpaintPipeline')
            print('The AutoPipelineForInpainting will be used')
            self.pipe = AutoPipelineForInpainting.from_pretrained(
                self.weights,
                torch_dtype=torch.float16,
                force_download=True, 
                resume_download=False,
                safety_checker = None
            ).to("cuda")
            
    @staticmethod        
    def generate_mask(aa_size, bbox, x_bias, y_bias):
        mask = np.full(shape=(aa_size, aa_size, 3), fill_value=[0,0,0], dtype=np.uint8)
        mask = Image.fromarray(mask)
        draw = ImageDraw.Draw(mask)
        
        bb_width = bbox[2] - bbox[0]
        bb_height = bbox[3] - bbox[1]
        
        x_top, y_top, x_bottom, y_bottom = (aa_size - bb_width)//2 - x_bias,(aa_size - bb_height)//2 - y_bias, (aa_size + bb_width)//2 - x_bias, (aa_size + bb_height)//2 - y_bias
        draw.rectangle([x_top, y_top, x_bottom, y_bottom], fill=(255, 255, 255))
        return mask.resize((512, 512)), [x_top, y_top, x_bottom, y_bottom]
        
    @staticmethod
    def generate_attention_area(img, bbox, increase_scale = 1.2, aa_size = None):
        img_width, img_height = img.size
        bb_width = bbox[2] - bbox[0]
        bb_height = bbox[3] - bbox[1]

        if aa_size is None:
            aa_size = int(round(max(bb_width, bb_height) * increase_scale))
        x_center, y_center = bbox[0] + (bb_width)//2, bbox[1] + (bb_height)//2
        x_top, y_top, x_bottom, y_bottom = max(0, x_center - aa_size//2), max(0,y_center - aa_size//2), min(img_width, x_center + aa_size//2), min(img_height, y_center + aa_size//2)
        x_diffs, y_diffs = [x_top - x_center + aa_size//2, x_center + aa_size//2 - x_bottom], [y_top - y_center + aa_size//2, y_center + aa_size//2 - y_bottom]
    
        if np.argmax(x_diffs)==0:
            x_bias = x_diffs[0]
            x_bottom += x_bias
        else:
            x_bias = -x_diffs[1]
            x_top += x_bias
    
        if np.argmax(y_diffs)==0:
            y_bias = y_diffs[0]
            y_bottom += y_bias
        else:
            y_bias = -y_diffs[1]
            y_top += y_bias

        mask, segment_area = AugmentationGenerator.generate_mask(aa_size, bbox, x_bias, y_bias)
        attention_area = img.crop([x_top, y_top, x_bottom, y_bottom]).resize((512,512), resample=Image.Resampling.BILINEAR)
        return attention_area, mask, segment_area, aa_size

    def __call__(self, guidance_scale: float = 10, num_inference_steps: int = 50, negative_prompt: str = None, bb_num: int = 1, increase_scale: float = 1.2, aa_size: int = None):
        ann_id = 0
        img_id = 0
        for img_file in self.img_files:
            img = Image.open(self.dir_images + img_file)
            img_info = {
                "id": img_id,
                "file_name": img_file,
                "height": img.size[1],
                "width": img.size[0]
            }
            self.coco["images"] += [img_info]
            
            for cls in self.classes:
                prompts = cls['prompts']
                category_id = cls['id']
                sample_bboxes = random.sample(self.bboxs, bb_num)
                for bbox in sample_bboxes:
                    prompt = random.choice(prompts)
                    attention_area, mask, segment_area, attention_area_size = AugmentationGenerator.generate_attention_area(img, bbox, increase_scale, aa_size)
                    bbox_w, bbox_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    bbox_area = bbox_w * bbox_h
                    
                    images = self.pipe(
                        prompt=prompt,
                        image=attention_area,
                        mask_image=mask,
                        negative_prompt=negative_prompt,
                        guidance_scale=guidance_scale,
                        num_images_per_prompt=1,
                        num_inference_steps=num_inference_steps
                    ).images
                    
                    img.paste(images[0].resize((attention_area_size, attention_area_size), resample=Image.Resampling.BILINEAR).crop(segment_area), bbox)
            
                    ann = {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": category_id,
                        "bbox": [bbox[0], bbox[1], bbox_w, bbox_h],
                        "area": bbox_area,
                        "segmentation": [],
                        "iscrowd": 0
                    }
                    self.coco["annotations"] += [ann]
                    
                    ann_id += 1
            img.save(self.dir_dataset + img_file)
            img_id += 1
        
        with open(self.final_json, 'w') as fp: # Final JSON file with COCO annotations
            json.dump(self.coco, fp)

def main():
    parser = argparse.ArgumentParser(description='Synthetic augmentation generator based on diffusion models.')
    parser.add_argument('--source_json', type=str, default=None, required=True, help="Path to JSON file with prompts and bboxes.")
    parser.add_argument('--final_json', type=str, default=None, required=True, help='Path to generated JSON file in COCO format.')
    parser.add_argument('--dir_images', type=str, default=None, required=True, help='Path to directory with original images.')
    parser.add_argument('--dir_dataset', type=str, default=None, required=True, help='Path to directory for augmented images.')
    parser.add_argument('--weights', type=str, required=False, default="stabilityai/stable-diffusion-2-inpainting", help='Path to directory with model. HuggingFace hub is supported.')
    
    parser.add_argument('--guidance_scale', type=float, required=False, default=10, help='A higher guidance scale value encourages the model to generate images closely linked to the text prompt at the expense of lower image quality. Guidance scale is enabled when guidance_scale > 1.')
    parser.add_argument('--num_inference_steps', type=int, required=False, default=50, help='The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.')
    parser.add_argument('--negative_prompt', type=str, required=False, default=None, help='The prompt to guide what to not include in image generation.')
    parser.add_argument('--bbox_number', type=int, required=False, default=1, help='The number of bboxes generated for each class in one image.')
    parser.add_argument('--increase_scale', type=float, required=False, default=1.2, help='This parameter is responsible for the value by which the size of the bbox is multiplied to allocate the attention area. This parameter must be greater than 1. This parameter will be ignored if parameter --aa_size has a value other than None')
    parser.add_argument('--aa_size', type=int, required=False, default=None, help='This parameter is responsible for the attention area size.')
    args = parser.parse_args()

    Generator = AugmentationGenerator(source_json=args.source_json, final_json=args.final_json, dir_images=args.dir_images, dir_dataset=args.dir_dataset, weights=args.weights)
    Generator(guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps, bb_num=args.bbox_number, negative_prompt=args.negative_prompt, increase_scale=args.increase_scale, aa_size=args.aa_size)
    
if __name__ == "__main__":
    main()