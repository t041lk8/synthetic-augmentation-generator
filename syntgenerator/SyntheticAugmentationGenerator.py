import os
import sys
import json
import math
import random

import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm.autonotebook import tqdm
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from datasets import load_from_disk

class InpaintDataset(torch.utils.data.Dataset):
    def __init__(self, 
                dataset_dir: str or os.PathLike, 
                tokenizer, 
                size: int=512):
        self.size = size
        self.tokenizer = tokenizer

        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise ValueError("Dataset doesn't exists.")

        self.dataset = load_from_disk(self.dataset_dir)
        self.images = self.dataset['images']
        self.prompts = self.dataset['text']
        self.masks = self.dataset['masks']
        self.instance_images_path = list(Path(dataset_dir).iterdir())
        self.num_instance_images = len(self.images)
        self._length = self.num_instance_images
        self.image_transforms_resize_and_crop = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                # transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            ]
        )

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image =self.images[index % self.num_instance_images]
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        instance_image = self.image_transforms_resize_and_crop(instance_image)

        example["PIL_images"] = instance_image
        example["instance_images"] = self.image_transforms(instance_image)

        example["instance_prompt_ids"] = self.tokenizer(
            self.prompts[index % self.num_instance_images],
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        example['masks'] = self.masks[index % self.num_instance_images]
        return example

class SDItrainer():
    def __init__(self, 
                pretrained_model_name_or_path: str or os.PathLike, 
                output_dir: str or os.PathLike):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.logging_dir = Path(self.output_dir, 'logs')
        self.project_config = ProjectConfiguration(
            project_dir=self.output_dir, logging_dir=self.logging_dir
        )

        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

    def __call__(self, 
                data_dir: str = None, 
                train_batch_size: int = 1, 
                max_train_steps: int = 400, 
                resolution: int = 512, 
                lr: float = 5e-6, 
                betas: tuple = (0.9, 0.999), 
                weight_decay: float = 1e-2, 
                eps: float = 1e-08, 
                gradient_accumulation_steps: int = 1,
                num_warmup_steps: int = 0,
                checkpoint_save: int = 500):
        optimizer_class = torch.optim.AdamW
        params_to_optimize = (
            self.unet.parameters()
        )
        self.optimizer = optimizer_class(
            params_to_optimize,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps
        )
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision="no",
            log_with="tensorboard",
            project_config=self.project_config,
        )
        self.lr_scheduler = get_scheduler(
            "constant",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps * self.accelerator.num_processes,
        )

        train_dataset = InpaintDataset(
            dataset_dir=data_dir,
            tokenizer=self.tokenizer,
            size=resolution,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=self.collate_fn
        )

        self.unet, self.optimizer, train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.unet, self.optimizer, train_dataloader, self.lr_scheduler)
        self.accelerator.register_for_checkpointing(self.lr_scheduler)
        weight_dtype = torch.float32

        self.vae.to(self.accelerator.device, dtype=weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=weight_dtype)
        self.accelerator.init_trackers("dreambooth")

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.accelerator.gradient_accumulation_steps)
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
        total_batch_size = train_batch_size * self.accelerator.num_processes * self.accelerator.gradient_accumulation_steps

        global_step = 0
        first_epoch = 0

        progress_bar = tqdm(range(global_step, max_train_steps))
        progress_bar.set_description("Steps")

        for epoch in range(first_epoch, num_train_epochs):
            self.unet.train()
            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    latents = self.vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor

                    masked_latents = self.vae.encode(
                        batch["masked_images"].reshape(batch["pixel_values"].shape).to(dtype=weight_dtype)
                    ).latent_dist.sample()
                    masked_latents = masked_latents * self.vae.config.scaling_factor

                    masks = batch["masks"]
                    mask = torch.stack(
                        [
                            torch.nn.functional.interpolate(mask, size=(resolution // 8, resolution // 8))
                            for mask in masks
                        ]
                    )
                    mask = mask.reshape(-1, 1, resolution // 8, resolution // 8)

                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()
                    
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                    latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)
                    encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
                    noise_pred = self.unet(latent_model_input, timesteps, encoder_hidden_states).sample

                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        params_to_clip = (
                            self.unet.parameters()
                        )
                        self.accelerator.clip_grad_norm_(params_to_clip, 1.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % checkpoint_save == 0:
                        if self.accelerator.is_main_process:
                            save_path = os.path.join(self.output_dir, f"checkpoint-{global_step}")
                            self.accelerator.save_state(save_path)

                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)

                if global_step >= max_train_steps:
                    break

            self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.pretrained_model_name_or_path,
                unet=self.accelerator.unwrap_model(self.unet),
                text_encoder=self.accelerator.unwrap_model(self.text_encoder),
            )
            pipeline.save_pretrained(self.output_dir)

        self.accelerator.end_training()
    
    def prepare_mask_and_masked_image(self, image, mask):
        image = np.array(image.convert("RGB"))
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        mask = np.array(mask.convert("L"))
        mask = mask.astype(np.float32) / 255.0
        mask = mask[None, None]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)
        masked_image = image * (mask < 0.5)

        return mask, masked_image

    def collate_fn(self, examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        masks = []
        masked_images = []
        for example in examples:
            pil_image = example["PIL_images"]
            mask = example["masks"]
            mask, masked_image = self.prepare_mask_and_masked_image(pil_image, mask)
            masks.append(mask)
            masked_images.append(masked_image)

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = self.tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids
        masks = torch.stack(masks)
        masked_images = torch.stack(masked_images)
        batch = {"input_ids": input_ids, "pixel_values": pixel_values, "masks": masks, "masked_images": masked_images}
        return batch

    

class AugmentationGenerator():
    def __init__(self, source_json: str or os.PathLike, 
                final_json: str or os.PathLike, 
                dir_images: str or os.PathLike, 
                dir_dataset: str or os.PathLike, 
                pipeline):
        self.dir_images = dir_images
        self.img_files = sorted(os.listdir(dir_images))
        with open(source_json) as f:
            annotations = json.load(f)
            self.bboxs = annotations['bboxs']
            self.classes = annotations['classes']
        self.pipe = pipeline

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

    def __call__(self, 
                guidance_scale: float = 10, 
                num_inference_steps: int = 50, 
                negative_prompt: str = None, 
                bb_num: int = 1, 
                increase_scale: float = 1.2, 
                aa_size: int = None):
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