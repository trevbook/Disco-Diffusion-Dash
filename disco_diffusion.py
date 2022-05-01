# This is an adaptation of Disco Diffusion; I'm allowing
# it to be run locally
# https://colab.research.google.com/drive/1sHfRn5Y0YKYKi1k-ifUSBFRNJ8_1sa39#scrollTo=FpZczxnOnPIU


# ====================================
#           PARSE USER INPUT
# ====================================

# Import statements
import argparse
from pathlib import Path
import json
import math
import json
import random
import pandas as pd
import numpy as np
import os
from time import sleep
from copy import deepcopy


# This method was taken from the original Disco Diffusion notebook; it'll split up some of the prompts
def split_prompts(prompts, max_frames):
    prompt_series = pd.Series([np.nan for a in range(max_frames)])
    for i, prompt in prompts.items():
        prompt_series[i] = prompt
    # prompt_series = prompt_series.astype(str)
    prompt_series = prompt_series.ffill().bfill()
    return prompt_series


# Setting up the Argument Parser
parser = argparse.ArgumentParser(description='Run DiscoDiffusion.')
parser.add_argument('--file', metavar='file_path', type=str, nargs=1,
                    help='The path to the config file for this Disco Diffusion run')
parser.add_argument('--dir', metavar='directory_path', type=str, nargs=1,
                    help='The path to a directory of config files for this Disco Diffusion run')
args = parser.parse_args()

# Declare the arg_dicts list; this is where we'll extract the argument dictionaries
arg_dicts = []

# We'll also keep a list of all of the different Paths we've pulled the arg_dicts from
config_paths = []

# Checking to see if the user provided the correct arguments
if (args.file is None and args.dir is None):
    print("You needed to supply either a path to a config file or a directory of config files.\n"
          "Exiting Disco Diffusion...")
    exit()

# If the user provided a file AND a directory, exit the file
elif (args.file is not None and args.dir is not None):
    print("You can't supply both a file path AND a directory path; choose just one of the two, and then restart.\n"
          "Exiting Disco Diffusion...")
    exit()

# Check to see if the user provided a valid file path
elif (args.file is not None and args.dir is None):
    file_path = Path(args.file[0])
    print(file_path)
    if (not file_path.exists()):
        print("You need to provide a valid file path. Please do that, and then try again.\n"
              "Exiting Disco Diffusion...")
        exit()

    # If the file_path DOES exist, we'll read in the JSON
    else:
        config_paths.append(file_path)
        with open(file_path, "r") as json_file:
            arg_dicts.append(json.load(json_file))

# Check to see if the user provided a valid directory
elif (args.file is None and args.dir is not None):
    dir_path = Path(args.dir[0])
    if (not dir_path.exists()):
        print("You need to provide a valid directory path. Please do that, and then try again.\n"
              "Exiting Disco Diffusion...")
        exit()

    # If the directory DOES exist, then we'll iterate through each of the files and read them in if they're .json
    for child_file in dir_path.iterdir():
        if (child_file.suffix == ".json"):
            config_paths.append(child_file)
            with open(child_file, "r") as jsonFile:
                arg_dicts.append(json.load(jsonFile))


# This method will create a version of either "prompts" dict with ints for keys (instead of strings)
def convert_prompt_dict_keys(prompt_dict):
    return {int(key): value for key, value in prompt_dict.items()}


# Keep a list of all of the unprocessed arg dicts
unprocessed_arg_dicts = deepcopy(arg_dicts)


# Now, we'll process the prompt lists from each of the arg_dicts
for idx, arg_dict in enumerate(arg_dicts):

    # Extract max_frames from the arg_dict
    max_frames = arg_dict["max_frames"]

    # Process the text_prompts

    text_prompts = convert_prompt_dict_keys(arg_dict["text_prompts"])
    prompts_series = split_prompts(text_prompts, max_frames) if text_prompts else None

    # Process the image_prompts
    image_prompts = convert_prompt_dict_keys(arg_dict["image_prompts"])
    image_prompts_series = split_prompts(image_prompts, max_frames) if image_prompts else None

    # Add these to the arg_dict
    arg_dicts[idx]["prompts_series"] = prompts_series
    arg_dicts[idx]["image_prompts_series"] = image_prompts_series

    # Evaluate cut_overview, cut_innercut, and cut_icgray_p
    arg_dicts[idx]["cut_overview"] = eval(arg_dicts[idx]["cut_overview"])
    arg_dicts[idx]["cut_innercut"] = eval(arg_dicts[idx]["cut_innercut"])
    arg_dicts[idx]["cut_icgray_p"] = eval(arg_dicts[idx]["cut_icgray_p"])

# ==========================================
# 1.2: PREPARE FOLDERS
# ==========================================


# @title 1.2 Prepare Folders
# @markdown If you connect your Google Drive, you can save the final image of each run on your drive.

google_drive = False  # @param {type:"boolean"}

# @markdown Click here if you'd like to save the diffusion model checkpoint file to (and/or load from) your Google Drive:
yes_please = False  # @param {type:"boolean"}

if google_drive is True:
    drive.mount('content/drive')
    root_path = 'content/drive/MyDrive/AI/Disco_Diffusion'
else:
    root_path = 'content'

import os
from os import path


# Simple create paths taken with modifications from Datamosh's Batch VQGAN+CLIP notebook
def createPath(filepath):
    if path.exists(filepath) == False:
        os.makedirs(filepath)

initDirPath = f'{root_path}/init_images'
createPath(initDirPath)
outDirPath = f'{root_path}/images_out'
createPath(outDirPath)

if google_drive and not yes_please or not google_drive:
    model_path = 'content/models'
    createPath(model_path)
if google_drive and yes_please:
    model_path = f'{root_path}/models'
    createPath(model_path)
# libraries = f'{root_path}/libraries'
# createPath(libraries)


# ==========================================
# 1.3: INSTALL AND IMPORT DEPENDENCIES 
# ==========================================


root_path = "content"
model_path = 'content/models'

model_256_downloaded = False
model_512_downloaded = False
model_secondary_downloaded = False

import sys
from dataclasses import dataclass
from functools import partial
import cv2
import pandas as pd
import gc
import io
import math
import timm
from IPython import display
import lpips
from PIL import Image, ImageOps
import requests
from glob import glob
import json
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm.notebook import tqdm
import clip
from resize_right import resize
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
from ipywidgets import Output
import hashlib
import ipywidgets as widgets
import os

sys.path.append('./ResizeRight')
sys.path.append('./CLIP')
sys.path.append('./guided-diffusion')
sys.path.append(".")
sys.path.append('./taming-transformers')
from taming.models import vqgan
from torchvision.datasets.utils import download_url
from functools import partial
from latent_diffusion.ldm.util import instantiate_from_config
from latent_diffusion.ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, \
    noise_like
from latent_diffusion.ldm.util import ismap
from numpy import asarray
from einops import rearrange, repeat
import torch, torchvision
import time
from omegaconf import OmegaConf
import warnings

# Ignore Warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if torch.cuda.get_device_capability(device) == (8, 0):  ## A100 fix thanks to Emad
    print('Disabling CUDNN for A100 gpu', file=sys.stderr)
    torch.backends.cudnn.enabled = False


# ==========================================
# 1.4: DEFINE NECESSARY FUNCTIONS
# ==========================================


# @title 1.4 Define necessary functions

# https://gist.github.com/adefossez/0646dbe9ed4005480a2407c62aac8869

def interp(t):
    return 3 * t ** 2 - 2 * t ** 3


def perlin(width, height, scale=10, device=None):
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)
    wx = 1 - interp(xs)
    wy = 1 - interp(ys)
    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))
    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)


def perlin_ms(octaves, width, height, grayscale, device=device):
    out_array = [0.5] if grayscale else [0.5, 0.5, 0.5]
    # out_array = [0.0] if grayscale else [0.0, 0.0, 0.0]
    for i in range(1 if grayscale else 3):
        scale = 2 ** len(octaves)
        oct_width = width
        oct_height = height
        for oct in octaves:
            p = perlin(oct_width, oct_height, scale, device)
            out_array[i] += p * oct
            scale //= 2
            oct_width *= 2
            oct_height *= 2
    return torch.cat(out_array)


def create_perlin_noise(octaves=[1, 1, 1, 1], width=2, height=2, grayscale=True):
    out = perlin_ms(octaves, width, height, grayscale)
    if grayscale:
        out = TF.resize(size=(side_y, side_x), img=out.unsqueeze(0))
        out = TF.to_pil_image(out.clamp(0, 1)).convert('RGB')
    else:
        out = out.reshape(-1, 3, out.shape[0] // 3, out.shape[1])
        out = TF.resize(size=(side_y, side_x), img=out)
        out = TF.to_pil_image(out.clamp(0, 1).squeeze())

    out = ImageOps.autocontrast(out)
    return out


def regen_perlin():
    if perlin_mode == 'color':
        init = create_perlin_noise([1.5 ** -i * 0.5 for i in range(12)], 1, 1, False)
        init2 = create_perlin_noise([1.5 ** -i * 0.5 for i in range(8)], 4, 4, False)
    elif perlin_mode == 'gray':
        init = create_perlin_noise([1.5 ** -i * 0.5 for i in range(12)], 1, 1, True)
        init2 = create_perlin_noise([1.5 ** -i * 0.5 for i in range(8)], 4, 4, True)
    else:
        init = create_perlin_noise([1.5 ** -i * 0.5 for i in range(12)], 1, 1, False)
        init2 = create_perlin_noise([1.5 ** -i * 0.5 for i in range(8)], 4, 4, True)

    init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1)
    del init2
    return init.expand(batch_size, -1, -1, -1)


def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def read_image_workaround(path):
    """OpenCV reads images as BGR, Pillow saves them as RGB. Work around
    this incompatibility to avoid colour inversions."""
    im_tmp = cv2.imread(path)
    return cv2.cvtColor(im_tmp, cv2.COLOR_BGR2RGB)


def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])


def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.reshape([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.reshape([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, skip_augs=False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.skip_augs = skip_augs
        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomPerspective(distortion_scale=0.4, p=0.7),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.15),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])

    def forward(self, input):
        input = T.Pad(input.shape[2] // 4, fill=0)(input)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)

        cutouts = []
        for ch in range(self.cutn):
            if ch > self.cutn - self.cutn // 4:
                cutout = input.clone()
            else:
                size = int(
                    max_size * torch.zeros(1, ).normal_(mean=.8, std=.3).clip(float(self.cut_size / max_size), 1.))
                offsetx = torch.randint(0, abs(sideX - size + 1), ())
                offsety = torch.randint(0, abs(sideY - size + 1), ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]

            if not self.skip_augs:
                cutout = self.augs(cutout)
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            del cutout

        cutouts = torch.cat(cutouts, dim=0)
        return cutouts


cutout_debug = False
padargs = {}


class MakeCutoutsDango(nn.Module):
    def __init__(self, cut_size,
                 Overview=4,
                 InnerCrop=0, IC_Size_Pow=0.5, IC_Grey_P=0.2
                 ):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P
        if args.animation_mode == 'None':
            self.augs = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomAffine(degrees=10, translate=(0.05, 0.05), interpolation=T.InterpolationMode.BILINEAR),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomGrayscale(p=0.1),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ])
        elif args.animation_mode == 'Video Input':
            self.augs = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomPerspective(distortion_scale=0.4, p=0.7),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomGrayscale(p=0.15),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ])
        elif args.animation_mode == '2D':
            self.augs = T.Compose([
                T.RandomHorizontalFlip(p=0.4),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomAffine(degrees=10, translate=(0.05, 0.05), interpolation=T.InterpolationMode.BILINEAR),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomGrayscale(p=0.1),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.3),
            ])

    def forward(self, input):
        cutouts = []
        gray = T.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        l_size = max(sideX, sideY)
        output_shape = [1, 3, self.cut_size, self.cut_size]
        output_shape_2 = [1, 3, self.cut_size + 2, self.cut_size + 2]
        pad_input = F.pad(input, (
            (sideY - max_size) // 2, (sideY - max_size) // 2, (sideX - max_size) // 2, (sideX - max_size) // 2), **padargs)
        cutout = resize(pad_input, out_shape=output_shape)

        if self.Overview > 0:
            if self.Overview <= 4:
                if self.Overview >= 1:
                    cutouts.append(cutout)
                if self.Overview >= 2:
                    cutouts.append(gray(cutout))
                if self.Overview >= 3:
                    cutouts.append(TF.hflip(cutout))
                if self.Overview == 4:
                    cutouts.append(gray(TF.hflip(cutout)))
            else:
                cutout = resize(pad_input, out_shape=output_shape)
                for _ in range(self.Overview):
                    cutouts.append(cutout)

            if cutout_debug:
                TF.to_pil_image(cutouts[0].clamp(0, 1).squeeze(0)).save("content/cutout_overview0.jpg", quality=99)

        if self.InnerCrop > 0:
            for i in range(self.InnerCrop):
                size = int(torch.rand([]) ** self.IC_Size_Pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)
            if cutout_debug:
                TF.to_pil_image(cutouts[-1].clamp(0, 1).squeeze(0)).save("content/cutout_InnerCrop.jpg", quality=99)
        cutouts = torch.cat(cutouts)
        if skip_augs is not True: cutouts = self.augs(cutouts)
        return cutouts


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff ** 2 + y_diff ** 2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


stop_on_next_loop = False  # Make sure GPU memory doesn't get corrupted from cancelling the run mid-way through, allow a full frame to complete


def do_run(args):

    print(f"Doing the run with the following arguments:\n{args}")

    seed = args.seed
    for frame_num in range(args.start_frame, args.max_frames):
        if stop_on_next_loop:
            break

        # display.clear_output(wait=True)

        # Print Frame progress if animation mode is on
        if args.animation_mode != "None":
            batchBar = tqdm(range(args.max_frames), desc="Frames")
            batchBar.n = frame_num
            batchBar.refresh()

        # Inits if not video frames
        if args.animation_mode != "Video Input":
            if args.init_image == '':
                init_image = None
            else:
                init_image = args.init_image
            init_scale = args.init_scale
            skip_steps = args.skip_steps

        if args.animation_mode == "2D":
            if args.key_frames:
                angle = args.angle_series[frame_num]
                zoom = args.zoom_series[frame_num]
                translation_x = args.translation_x_series[frame_num]
                translation_y = args.translation_y_series[frame_num]
                print(
                    f'angle: {angle}',
                    f'zoom: {zoom}',
                    f'translation_x: {translation_x}',
                    f'translation_y: {translation_y}',
                )

            if frame_num > 0:
                seed = seed + 1
                if resume_run and frame_num == args.start_frame:
                    img_0 = cv2.imread(batchFolder + f"/{args.batch_name}({args.batchNum})_{args.start_frame - 1:04}.png")
                else:
                    img_0 = cv2.imread('prevFrame.png')
                center = (1 * img_0.shape[1] // 2, 1 * img_0.shape[0] // 2)
                trans_mat = np.float32(
                    [[1, 0, translation_x],
                     [0, 1, translation_y]]
                )
                rot_mat = cv2.getRotationMatrix2D(center, angle, zoom)
                trans_mat = np.vstack([trans_mat, [0, 0, 1]])
                rot_mat = np.vstack([rot_mat, [0, 0, 1]])
                transformation_matrix = np.matmul(rot_mat, trans_mat)
                img_0 = cv2.warpPerspective(
                    img_0,
                    transformation_matrix,
                    (img_0.shape[1], img_0.shape[0]),
                    borderMode=cv2.BORDER_WRAP
                )
                cv2.imwrite('prevFrameScaled.png', img_0)
                init_image = 'prevFrameScaled.png'
                init_scale = args.frames_scale
                skip_steps = args.calc_frames_skip_steps

        if args.animation_mode == "Video Input":
            seed = seed + 1
            init_image = f'{videoFramesFolder}/{frame_num + 1:04}.jpg'
            init_scale = args.frames_scale
            skip_steps = args.calc_frames_skip_steps

        loss_values = []

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

        target_embeds, weights = [], []

        if args.prompts_series is not None and frame_num >= len(args.prompts_series):
            frame_prompt = args.prompts_series[-1]
        elif args.prompts_series is not None:
            frame_prompt = args.prompts_series[frame_num]
        else:
            frame_prompt = []

        if args.image_prompts_series is not None and frame_num >= len(args.image_prompts_series):
            image_prompt = args.image_prompts_series[-1]
        elif args.image_prompts_series is not None:
            image_prompt = args.image_prompts_series[frame_num]
        else:
            image_prompt = []

        model_stats = []
        for clip_model in clip_models:
            cutn = 16
            model_stat = {"clip_model": None, "target_embeds": [], "make_cutouts": None, "weights": []}
            model_stat["clip_model"] = clip_model

            for prompt in frame_prompt:
                txt, weight = parse_prompt(prompt)
                txt = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()

                if args.fuzzy_prompt:
                    for i in range(25):
                        model_stat["target_embeds"].append(
                            (txt + torch.randn(txt.shape).cuda() * args.rand_mag).clamp(0, 1))
                        model_stat["weights"].append(weight)
                else:
                    model_stat["target_embeds"].append(txt)
                    model_stat["weights"].append(weight)

            if image_prompt:
                model_stat["make_cutouts"] = MakeCutouts(clip_model.visual.input_resolution, cutn, skip_augs=skip_augs)
                for prompt in image_prompt:
                    path, weight = parse_prompt(prompt)
                    img = Image.open(fetch(path)).convert('RGB')
                    img = TF.resize(img, min(side_x, side_y, *img.size), T.InterpolationMode.LANCZOS)
                    batch = model_stat["make_cutouts"](TF.to_tensor(img).to(device).unsqueeze(0).mul(2).sub(1))
                    embed = clip_model.encode_image(normalize(batch)).float()
                    if fuzzy_prompt:
                        for i in range(25):
                            model_stat["target_embeds"].append(
                                (embed + torch.randn(embed.shape).cuda() * rand_mag).clamp(0, 1))
                            weights.extend([weight / cutn] * cutn)
                    else:
                        model_stat["target_embeds"].append(embed)
                        model_stat["weights"].extend([weight / cutn] * cutn)

            model_stat["target_embeds"] = torch.cat(model_stat["target_embeds"])
            model_stat["weights"] = torch.tensor(model_stat["weights"], device=device)
            if model_stat["weights"].sum().abs() < 1e-3:
                raise RuntimeError('The weights must not sum to 0.')
            model_stat["weights"] /= model_stat["weights"].sum().abs()
            model_stats.append(model_stat)

        init = None
        if init_image is not None:
            init = Image.open(fetch(init_image)).convert('RGB')
            init = init.resize((args.side_x, args.side_y), Image.LANCZOS)
            init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

        if args.perlin_init:
            if args.perlin_mode == 'color':
                init = create_perlin_noise([1.5 ** -i * 0.5 for i in range(12)], 1, 1, False)
                init2 = create_perlin_noise([1.5 ** -i * 0.5 for i in range(8)], 4, 4, False)
            elif args.perlin_mode == 'gray':
                init = create_perlin_noise([1.5 ** -i * 0.5 for i in range(12)], 1, 1, True)
                init2 = create_perlin_noise([1.5 ** -i * 0.5 for i in range(8)], 4, 4, True)
            else:
                init = create_perlin_noise([1.5 ** -i * 0.5 for i in range(12)], 1, 1, False)
                init2 = create_perlin_noise([1.5 ** -i * 0.5 for i in range(8)], 4, 4, True)
            # init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device)
            init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1)
            del init2

        cur_t = None

        def cond_fn(x, t, y=None):
            with torch.enable_grad():
                x_is_NaN = False
                x = x.detach().requires_grad_()
                n = x.shape[0]
                if use_secondary_model is True:
                    alpha = torch.tensor(diffusion.sqrt_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
                    sigma = torch.tensor(diffusion.sqrt_one_minus_alphas_cumprod[cur_t], device=device,
                                         dtype=torch.float32)
                    cosine_t = alpha_sigma_to_t(alpha, sigma)
                    out = secondary_model(x, cosine_t[None].repeat([n])).pred
                    fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                    x_in = out * fac + x * (1 - fac)
                    x_in_grad = torch.zeros_like(x_in)
                else:
                    my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
                    out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={'y': y})
                    fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                    x_in = out['pred_xstart'] * fac + x * (1 - fac)
                    x_in_grad = torch.zeros_like(x_in)
                for model_stat in model_stats:
                    for i in range(args.cutn_batches):
                        t_int = int(t.item()) + 1  # errors on last step without +1, need to find source
                        # when using SLIP Base model the dimensions need to be hard coded to avoid AttributeError: 'VisionTransformer' object has no attribute 'input_resolution'
                        try:
                            input_resolution = model_stat["clip_model"].visual.input_resolution
                        except:
                            input_resolution = 224

                        cuts = MakeCutoutsDango(input_resolution,
                                                Overview=args.cut_overview[1000 - t_int],
                                                InnerCrop=args.cut_innercut[1000 - t_int], IC_Size_Pow=args.cut_ic_pow,
                                                IC_Grey_P=args.cut_icgray_p[1000 - t_int]
                                                )
                        clip_in = normalize(cuts(x_in.add(1).div(2)))
                        image_embeds = model_stat["clip_model"].encode_image(clip_in).float()
                        dists = spherical_dist_loss(image_embeds.unsqueeze(1), model_stat["target_embeds"].unsqueeze(0))
                        dists = dists.view([args.cut_overview[1000 - t_int] + args.cut_innercut[1000 - t_int], n, -1])
                        losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
                        loss_values.append(losses.sum().item())  # log loss, probably shouldn't do per cutn_batch
                        x_in_grad += torch.autograd.grad(losses.sum() * clip_guidance_scale, x_in)[0] / cutn_batches
                tv_losses = tv_loss(x_in)
                if use_secondary_model is True:
                    range_losses = range_loss(out)
                else:
                    range_losses = range_loss(out['pred_xstart'])
                sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
                loss = tv_losses.sum() * tv_scale + range_losses.sum() * range_scale + sat_losses.sum() * sat_scale
                if init is not None and args.init_scale:
                    init_losses = lpips_model(x_in, init)
                    loss = loss + init_losses.sum() * args.init_scale
                x_in_grad += torch.autograd.grad(loss, x_in)[0]
                if torch.isnan(x_in_grad).any() == False:
                    grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
                else:
                    # print("NaN'd")
                    x_is_NaN = True
                    grad = torch.zeros_like(x)
            if args.clamp_grad and x_is_NaN == False:
                magnitude = grad.square().mean().sqrt()
                return grad * magnitude.clamp(max=args.clamp_max) / magnitude  # min=-0.02, min=-clamp_max, 
            return grad

        if model_config['timestep_respacing'].startswith('ddim'):
            sample_fn = diffusion.ddim_sample_loop_progressive
        else:
            sample_fn = diffusion.p_sample_loop_progressive

        image_display = Output()
        for i in range(args.n_batches):
            if args.animation_mode == 'None':
                batchBar = tqdm(range(args.n_batches), desc="Batches")
                batchBar.n = i
                batchBar.refresh()
            # print('')
            # display.display(image_display)
            gc.collect()
            torch.cuda.empty_cache()
            cur_t = diffusion.num_timesteps - skip_steps - 1
            total_steps = cur_t

            if perlin_init:
                init = regen_perlin()

            if model_config['timestep_respacing'].startswith('ddim'):
                samples = sample_fn(
                    model,
                    (batch_size, 3, args.side_y, args.side_x),
                    clip_denoised=clip_denoised,
                    model_kwargs={},
                    cond_fn=cond_fn,
                    progress=True,
                    skip_timesteps=skip_steps,
                    init_image=init,
                    randomize_class=randomize_class,
                    eta=eta,
                )
            else:
                samples = sample_fn(
                    model,
                    (batch_size, 3, args.side_y, args.side_x),
                    clip_denoised=clip_denoised,
                    model_kwargs={},
                    cond_fn=cond_fn,
                    progress=True,
                    skip_timesteps=skip_steps,
                    init_image=init,
                    randomize_class=randomize_class,
                )

            # with run_display:
            # display.clear_output(wait=True)
            imgToSharpen = None
            for j, sample in enumerate(samples):
                cur_t -= 1
                intermediateStep = False
                if args.steps_per_checkpoint is not None:
                    if j % args.steps_per_checkpoint == 0 and j > 0:
                        intermediateStep = True
                elif j in args.intermediate_saves:
                    intermediateStep = True
                with image_display:
                    if j % args.display_rate == 0 or cur_t == -1 or intermediateStep == True:
                        for k, image in enumerate(sample['pred_xstart']):

                            # tqdm.write(f'Batch {i}, step {j}, output {k}:')
                            current_time = datetime.now().strftime('%y%m%d-%H%M%S_%f')
                            percent = math.ceil(j / total_steps * 100)
                            if args.n_batches > 0:
                                # if intermediates are saved to the subfolder, don't append a step or percentage to the name
                                if cur_t == -1 and args.intermediates_in_subfolder is True:
                                    save_num = f'{frame_num:04}' if animation_mode != "None" else i
                                    filename = f'{args.batch_name}({i})_{save_num}.png'
                                else:
                                    # If we're working with percentages, append it
                                    if args.steps_per_checkpoint is not None:
                                        filename = f'{args.batch_name}({i})_{i:04}-{percent:02}%.png'
                                    # Or else, iIf we're working with specific steps, append those
                                    else:
                                        filename = f'{args.batch_name}({i})_{i:04}-{j:03}.png'
                            image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                            if j % args.display_rate == 0 or cur_t == -1:
                                image.save('progress.png')
                                # display.clear_output(wait=True)
                                # display.display(display.Image('progress.png'))
                            if args.steps_per_checkpoint is not None:
                                if j % args.steps_per_checkpoint == 0 and j > 0:
                                    if args.intermediates_in_subfolder is True:
                                        image.save(f'{partialFolder}/{filename}')
                                    else:
                                        image.save(f'{batchFolder}/{filename}')
                            else:
                                if j in args.intermediate_saves:
                                    if args.intermediates_in_subfolder is True:
                                        image.save(f'{partialFolder}/{filename}')
                                    else:
                                        image.save(f'{batchFolder}/{filename}')
                            if cur_t == -1:
                                if frame_num == 0:
                                    save_settings(args, i)
                                if args.animation_mode != "None":
                                    image.save('prevFrame.png')
                                if args.sharpen_preset != "Off" and animation_mode == "None":
                                    imgToSharpen = image
                                    if args.keep_unsharp is True:
                                        image.save(f'{unsharpenFolder}/{filename}')
                                else:
                                    image.save(f'{batchFolder}/{filename}')
                                # if frame_num != args.max_frames-1:
                                #   display.clear_output()

            with image_display:
                if args.sharpen_preset != "Off" and animation_mode == "None":
                    print('Starting Diffusion Sharpening...')
                    do_superres(imgToSharpen, f'{batchFolder}/{filename}')
                    # display.clear_output()

            plt.plot(np.array(loss_values), 'r')

# TODO: Clean up the save_settings() so that it actually just copies the config file 
def save_settings(args, cur_batch_num):
    # setting_list = {
    #     'text_prompts': args.text_prompts,
    #     'image_prompts': args.image_prompts,
    #     'clip_guidance_scale': args.clip_guidance_scale,
    #     'tv_scale': args.tv_scale,
    #     'range_scale': args.range_scale,
    #     'sat_scale': args.sat_scale,
    #     'cutn_batches': args.cutn_batches,
    #     'max_frames': args.max_frames,
    #     'interp_spline': args.interp_spline,
    #     'init_image': args.init_image,
    #     'init_scale': args.init_scale,
    #     'skip_steps': args.skip_steps,
    #     'frames_scale': args.frames_scale,
    #     'frames_skip_steps': args.frames_skip_steps,
    #     'perlin_init': args.perlin_init,
    #     'perlin_mode': args.perlin_mode,
    #     'skip_augs': args.skip_augs,
    #     'randomize_class': randomize_class,
    #     'clip_denoised': clip_denoised,
    #     'clamp_grad': clamp_grad,
    #     'clamp_max': clamp_max,
    #     'seed': seed,
    #     'fuzzy_prompt': fuzzy_prompt,
    #     'rand_mag': rand_mag,
    #     'eta': eta,
    #     'width': width_height[0],
    #     'height': width_height[1],
    #     'diffusion_model': diffusion_model,
    #     'use_secondary_model': use_secondary_model,
    #     'steps': steps,
    #     'diffusion_steps': diffusion_steps,
    #     'ViTB32': ViTB32,
    #     'ViTB16': ViTB16,
    #     'ViTL14': ViTL14,
    #     'RN101': RN101,
    #     'RN50': RN50,
    #     'RN50x4': RN50x4,
    #     'RN50x16': RN50x16,
    #     'RN50x64': RN50x64,
    #     'cut_overview': str(cut_overview),
    #     'cut_innercut': str(cut_innercut),
    #     'cut_ic_pow': cut_ic_pow,
    #     'cut_icgray_p': str(cut_icgray_p),
    #     'key_frames': key_frames,
    #     'max_frames': max_frames,
    #     'angle': angle,
    #     'zoom': zoom,
    #     'translation_x': translation_x,
    #     'translation_y': translation_y,
    #     'video_init_path': video_init_path,
    #     'extract_nth_frame': extract_nth_frame,
    # }
    settings_list = {
        'text_prompts': args.text_prompts,
        'image_prompts': args.image_prompts,
        "seed": args.seed
    }

    # print('Settings:', setting_list)
    with open(f"{batchFolder}/{args.batch_name}({cur_batch_num})_settings.txt", "w+", encoding="utf-8") as f:  # save settings
        json.dump(settings_list, f, ensure_ascii=False, indent=4)


# ==========================================
# 1.5: DEFINE THE SECONDARY DIFFUSION MODEL
# ==========================================

# @title 1.5 Define the secondary diffusion model

def append_dims(x, n):
    return x[(Ellipsis, *(None,) * (n - x.ndim))]


def expand_to_planes(x, shape):
    return append_dims(x, len(shape)).repeat([1, 1, *shape[2:]])


def alpha_sigma_to_t(alpha, sigma):
    return torch.atan2(sigma, alpha) * 2 / math.pi


def t_to_alpha_sigma(t):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


@dataclass
class DiffusionOutput:
    v: torch.Tensor
    pred: torch.Tensor
    eps: torch.Tensor


class ConvBlock(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
        )


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class SecondaryDiffusionImageNet(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count

        self.timestep_embed = FourierFeatures(1, 16)

        self.net = nn.Sequential(
            ConvBlock(3 + 16, c),
            ConvBlock(c, c),
            SkipBlock([
                nn.AvgPool2d(2),
                ConvBlock(c, c * 2),
                ConvBlock(c * 2, c * 2),
                SkipBlock([
                    nn.AvgPool2d(2),
                    ConvBlock(c * 2, c * 4),
                    ConvBlock(c * 4, c * 4),
                    SkipBlock([
                        nn.AvgPool2d(2),
                        ConvBlock(c * 4, c * 8),
                        ConvBlock(c * 8, c * 4),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    ]),
                    ConvBlock(c * 8, c * 4),
                    ConvBlock(c * 4, c * 2),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ]),
                ConvBlock(c * 4, c * 2),
                ConvBlock(c * 2, c),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ]),
            ConvBlock(c * 2, c),
            nn.Conv2d(c, 3, 3, padding=1),
        )

    def forward(self, input, t):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        v = self.net(torch.cat([input, timestep_embed], dim=1))
        alphas, sigmas = map(partial(append_dims, n=v.ndim), t_to_alpha_sigma(t))
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)


class SecondaryDiffusionImageNet2(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count
        cs = [c, c * 2, c * 2, c * 4, c * 4, c * 8]

        self.timestep_embed = FourierFeatures(1, 16)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.net = nn.Sequential(
            ConvBlock(3 + 16, cs[0]),
            ConvBlock(cs[0], cs[0]),
            SkipBlock([
                self.down,
                ConvBlock(cs[0], cs[1]),
                ConvBlock(cs[1], cs[1]),
                SkipBlock([
                    self.down,
                    ConvBlock(cs[1], cs[2]),
                    ConvBlock(cs[2], cs[2]),
                    SkipBlock([
                        self.down,
                        ConvBlock(cs[2], cs[3]),
                        ConvBlock(cs[3], cs[3]),
                        SkipBlock([
                            self.down,
                            ConvBlock(cs[3], cs[4]),
                            ConvBlock(cs[4], cs[4]),
                            SkipBlock([
                                self.down,
                                ConvBlock(cs[4], cs[5]),
                                ConvBlock(cs[5], cs[5]),
                                ConvBlock(cs[5], cs[5]),
                                ConvBlock(cs[5], cs[4]),
                                self.up,
                            ]),
                            ConvBlock(cs[4] * 2, cs[4]),
                            ConvBlock(cs[4], cs[3]),
                            self.up,
                        ]),
                        ConvBlock(cs[3] * 2, cs[3]),
                        ConvBlock(cs[3], cs[2]),
                        self.up,
                    ]),
                    ConvBlock(cs[2] * 2, cs[2]),
                    ConvBlock(cs[2], cs[1]),
                    self.up,
                ]),
                ConvBlock(cs[1] * 2, cs[1]),
                ConvBlock(cs[1], cs[0]),
                self.up,
            ]),
            ConvBlock(cs[0] * 2, cs[0]),
            nn.Conv2d(cs[0], 3, 3, padding=1),
        )

    def forward(self, input, t):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        v = self.net(torch.cat([input, timestep_embed], dim=1))
        alphas, sigmas = map(partial(append_dims, n=v.ndim), t_to_alpha_sigma(t))
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)


# ==========================================
# 2.1: Diffusion and CLIP Model Settings
# ==========================================

# @markdown ####**Models Settings:**
diffusion_model = "512x512_diffusion_uncond_finetune_008100"  # @param ["256x256_diffusion_uncond", "512x512_diffusion_uncond_finetune_008100"]
use_secondary_model = True  # @param {type: 'boolean'}

timestep_respacing = '50'  # param ['25','50','100','150','250','500','1000','ddim25','ddim50', 'ddim75', 'ddim100','ddim150','ddim250','ddim500','ddim1000']  
diffusion_steps = 1000  # param {type: 'number'}
use_checkpoint = True  # @param {type: 'boolean'}
ViTB32 = True  # @param{type:"boolean"}
ViTB16 = True  # @param{type:"boolean"}
ViTL14 = False  # @param{type:"boolean"}
RN101 = False  # @param{type:"boolean"}
RN50 = True  # @param{type:"boolean"}
RN50x4 = False  # @param{type:"boolean"}
RN50x16 = False  # @param{type:"boolean"}
RN50x64 = False  # @param{type:"boolean"}
SLIPB16 = False  # param{type:"boolean"}
SLIPL16 = False  # param{type:"boolean"}

# @markdown If you're having issues with model downloads, check this to compare SHA's:
check_model_SHA = False  # @param{type:"boolean"}

model_256_SHA = '983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a'
model_512_SHA = '9c111ab89e214862b76e1fa6a1b3f1d329b1a88281885943d2cdbe357ad57648'
model_secondary_SHA = '983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a'

model_256_link = 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt'
model_512_link = 'https://v-diffusion.s3.us-west-2.amazonaws.com/512x512_diffusion_uncond_finetune_008100.pt'
model_secondary_link = 'https://v-diffusion.s3.us-west-2.amazonaws.com/secondary_model_imagenet_2.pth'

model_256_path = f'{model_path}/256x256_diffusion_uncond.pt'
model_512_path = f'{model_path}/512x512_diffusion_uncond_finetune_008100.pt'
model_secondary_path = f'{model_path}/secondary_model_imagenet_2.pth'

# Download the diffusion model
if diffusion_model == '256x256_diffusion_uncond':
    if os.path.exists(model_256_path) and check_model_SHA:
        print('Checking 256 Diffusion File')
        with open(model_256_path, "rb") as f:
            bytes = f.read()
            hash = hashlib.sha256(bytes).hexdigest();
        if hash == model_256_SHA:
            print('256 Model SHA matches')
            model_256_downloaded = True
        else:
            print("256 Model SHA doesn't match, redownloading...")
            model_256_downloaded = True
    elif os.path.exists(model_256_path) and not check_model_SHA or model_256_downloaded == True:
        pass
    else:
        model_256_downloaded = True
elif diffusion_model == '512x512_diffusion_uncond_finetune_008100':
    if os.path.exists(model_512_path) and check_model_SHA:
        print('Checking 512 Diffusion File')
        with open(model_512_path, "rb") as f:
            bytes = f.read()
            hash = hashlib.sha256(bytes).hexdigest();
        if hash == model_512_SHA:
            print('512 Model SHA matches')
            model_512_downloaded = True
        else:
            print("512 Model SHA doesn't match, redownloading...")
            model_512_downloaded = True
    elif os.path.exists(model_512_path) and not check_model_SHA or model_512_downloaded == True:
        print('512 Model already downloaded, check check_model_SHA if the file is corrupt')
    else:
        model_512_downloaded = True

# Download the secondary diffusion model v2
if use_secondary_model == True:
    if os.path.exists(model_secondary_path) and check_model_SHA:
        print('Checking Secondary Diffusion File')
        with open(model_secondary_path, "rb") as f:
            bytes = f.read()
            hash = hashlib.sha256(bytes).hexdigest();
        if hash == model_secondary_SHA:
            print('Secondary Model SHA matches')
            model_secondary_downloaded = True
        else:
            print("Secondary Model SHA doesn't match, redownloading...")
            model_secondary_downloaded = True
    elif os.path.exists(model_secondary_path) and not check_model_SHA or model_secondary_downloaded == True:
        print('Secondary Model already downloaded, check check_model_SHA if the file is corrupt')
    else:
        model_secondary_downloaded = True

model_config = model_and_diffusion_defaults()


secondary_model_ver = 2
model_default = model_config['image_size']

if secondary_model_ver == 2:
    secondary_model = SecondaryDiffusionImageNet2()
    secondary_model.load_state_dict(torch.load(f'{model_path}/secondary_model_imagenet_2.pth', map_location='cpu'))
secondary_model.eval().requires_grad_(False).to(device)

clip_models = []
if ViTB32 is True: clip_models.append(clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(device))
if ViTB16 is True: clip_models.append(clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to(device))
if ViTL14 is True: clip_models.append(clip.load('ViT-L/14', jit=False)[0].eval().requires_grad_(False).to(device))
if RN50 is True: clip_models.append(clip.load('RN50', jit=False)[0].eval().requires_grad_(False).to(device))
if RN50x4 is True: clip_models.append(clip.load('RN50x4', jit=False)[0].eval().requires_grad_(False).to(device))
if RN50x16 is True: clip_models.append(clip.load('RN50x16', jit=False)[0].eval().requires_grad_(False).to(device))
if RN50x64 is True: clip_models.append(clip.load('RN50x64', jit=False)[0].eval().requires_grad_(False).to(device))
if RN101 is True: clip_models.append(clip.load('RN101', jit=False)[0].eval().requires_grad_(False).to(device))

if SLIPB16:
    SLIPB16model = SLIP_VITB16(ssl_mlp_dim=4096, ssl_emb_dim=256)
    sd = torch.load(f'{model_path}/slip_base_100ep.pt')
    real_sd = {}
    for k, v in sd['state_dict'].items():
        real_sd['.'.join(k.split('.')[1:])] = v
    del sd
    SLIPB16model.load_state_dict(real_sd)
    SLIPB16model.requires_grad_(False).eval().to(device)

    clip_models.append(SLIPB16model)

if SLIPL16:
    SLIPL16model = SLIP_VITL16(ssl_mlp_dim=4096, ssl_emb_dim=256)
    sd = torch.load(f'{model_path}/slip_large_100ep.pt')
    real_sd = {}
    for k, v in sd['state_dict'].items():
        real_sd['.'.join(k.split('.')[1:])] = v
    del sd
    SLIPL16model.load_state_dict(real_sd)
    SLIPL16model.requires_grad_(False).eval().to(device)

    clip_models.append(SLIPL16model)

normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
lpips_model = lpips.LPIPS(net='vgg').to(device)

# 3.1: SETTINGS

# @markdown ####**Basic Settings:**
batch_name = 'Get Better Lyrics'  # @param{type: 'string'}
steps = 320  # @param [25,50,100,150,250,500,1000]{type: 'raw', allow-input: true} # CHANGE THE STEPS
width_height = [1150, 700]  # LANDSCAPE BIG
# width_height = [600, 400] # LANDSCAPE SMALL
# width_height = [512, 1088] # PHONE WALLPAPER
clip_guidance_scale = 5000  # @param{type: 'number'}
tv_scale = 0  # @param{type: 'number'}
range_scale = 150  # @param{type: 'number'}
sat_scale = 0  # @param{type: 'number'}
cutn_batches = 4  # @param{type: 'number'}
skip_augs = False  # @param{type: 'boolean'}

# @markdown ---

# @markdown ####**Init Settings:**
init_image = None
# init_image = "content/init_images/colored question mark.png" #@param{type: 'string'} # CHANGE THE INIT IMAGE
init_scale = 1  # @param{type: 'integer'}
skip_steps = 0
# skip_steps = int(steps/2) #@param{type: 'integer'}
# @markdown *Make sure you set skip_steps to ~50% of your steps if you want to use an init image.*

# Get corrected sizes
side_x = (width_height[0] // 64) * 64;
side_y = (width_height[1] // 64) * 64;
if side_x != width_height[0] or side_y != width_height[1]:
    print(f'Changing output size to {side_x}x{side_y}. Dimensions must by multiples of 64.')

# Update Model Settings
timestep_respacing = f'ddim{steps}'
diffusion_steps = (1000 // steps) * steps if steps < 1000 else steps




# 3.2: ANIMATION SETTINGS

# @markdown ####**Animation Mode:**
animation_mode = "None"  # @param['None', '2D', 'Video Input']
# @markdown *For animation, you probably want to turn `cutn_batches` to 1 to make it quicker.*


# @markdown ---

# @markdown ####**Video Input Settings:**
video_init_path = "content/training.mp4"  # @param {type: 'string'}
extract_nth_frame = 2  # @param {type:"number"} 

# if animation_mode == "Video Input":
#   videoFramesFolder = f'content/videoFrames'
#   createPath(videoFramesFolder)
#   print(f"Exporting Video Frames (1 every {extract_nth_frame})...")
#   try:
#     !rm {videoFramesFolder}/*.jpg
#   except:
#     print('')
#   vf = f'"select=not(mod(n\,{extract_nth_frame}))"'
#   !ffmpeg -i {video_init_path} -vf {vf} -vsync vfr -q:v 2 -loglevel error -stats {videoFramesFolder}/%04d.jpg


# @markdown ---

# @markdown ####**2D Animation Settings:**
# @markdown `zoom` is a multiplier of dimensions, 1 is no zoom.

key_frames = True  # @param {type:"boolean"}
max_frames = 10000  # @param {type:"number"}

if animation_mode == "Video Input":
    max_frames = len(glob(f'{videoFramesFolder}/*.jpg'))

interp_spline = 'Linear'  # Do not change, currently will not look good. param ['Linear','Quadratic','Cubic']{type:"string"}
angle = "0:(0)"  # @param {type:"string"}
zoom = "0: (1), 10: (1.05)"  # @param {type:"string"}
translation_x = "0: (0)"  # @param {type:"string"}
translation_y = "0: (0)"  # @param {type:"string"}

# @markdown ---

# @markdown ####**Coherency Settings:**
# @markdown `frame_scale` tries to guide the new frame to looking like the old one. A good default is 1500.
frames_scale = 1500  # @param{type: 'integer'}
# @markdown `frame_skip_steps` will blur the previous frame - higher values will flicker less but struggle to add enough new detail to zoom into.
frames_skip_steps = '60%'  # @param ['40%', '50%', '60%', '70%', '80%'] {type: 'string'}


def parse_key_frames(string, prompt_parser=None):
    """Given a string representing frame numbers paired with parameter values at that frame,
    return a dictionary with the frame numbers as keys and the parameter values as the values.

    Parameters
    ----------
    string: string
        Frame numbers paired with parameter values at that frame number, in the format
        'framenumber1: (parametervalues1), framenumber2: (parametervalues2), ...'
    prompt_parser: function or None, optional
        If provided, prompt_parser will be applied to each string of parameter values.

    Returns
    -------
    dict
        Frame numbers as keys, parameter values at that frame number as values

    Raises
    ------
    RuntimeError
        If the input string does not match the expected format.

    Examples
    --------
    >>> parse_key_frames("10:(Apple: 1| Orange: 0), 20: (Apple: 0| Orange: 1| Peach: 1)")
    {10: 'Apple: 1| Orange: 0', 20: 'Apple: 0| Orange: 1| Peach: 1'}

    >>> parse_key_frames("10:(Apple: 1| Orange: 0), 20: (Apple: 0| Orange: 1| Peach: 1)", prompt_parser=lambda x: x.lower()))
    {10: 'apple: 1| orange: 0', 20: 'apple: 0| orange: 1| peach: 1'}
    """
    import re
    pattern = r'((?P<frame>[0-9]+):[\s]*[\(](?P<param>[\S\s]*?)[\)])'
    frames = dict()
    for match_object in re.finditer(pattern, string):
        frame = int(match_object.groupdict()['frame'])
        param = match_object.groupdict()['param']
        if prompt_parser:
            frames[frame] = prompt_parser(param)
        else:
            frames[frame] = param

    if frames == {} and len(string) != 0:
        raise RuntimeError('Key Frame string not correctly formatted')
    return frames


def get_inbetweens(key_frames, integer=False):
    """Given a dict with frame numbers as keys and a parameter value as values,
    return a pandas Series containing the value of the parameter at every frame from 0 to max_frames.
    Any values not provided in the input dict are calculated by linear interpolation between
    the values of the previous and next provided frames. If there is no previous provided frame, then
    the value is equal to the value of the next provided frame, or if there is no next provided frame,
    then the value is equal to the value of the previous provided frame. If no frames are provided,
    all frame values are NaN.

    Parameters
    ----------
    key_frames: dict
        A dict with integer frame numbers as keys and numerical values of a particular parameter as values.
    integer: Bool, optional
        If True, the values of the output series are converted to integers.
        Otherwise, the values are floats.

    Returns
    -------
    pd.Series
        A Series with length max_frames representing the parameter values for each frame.

    Examples
    --------
    >>> max_frames = 5
    >>> get_inbetweens({1: 5, 3: 6})
    0    5.0
    1    5.0
    2    5.5
    3    6.0
    4    6.0
    dtype: float64

    >>> get_inbetweens({1: 5, 3: 6}, integer=True)
    0    5
    1    5
    2    5
    3    6
    4    6
    dtype: int64
    """
    key_frame_series = pd.Series([np.nan for a in range(max_frames)])

    for i, value in key_frames.items():
        key_frame_series[i] = value
    key_frame_series = key_frame_series.astype(float)

    interp_method = interp_spline

    if interp_method == 'Cubic' and len(key_frames.items()) <= 3:
        interp_method = 'Quadratic'

    if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
        interp_method = 'Linear'

    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[max_frames - 1] = key_frame_series[key_frame_series.last_valid_index()]
    # key_frame_series = key_frame_series.interpolate(method=intrp_method,order=1, limit_direction='both')
    key_frame_series = key_frame_series.interpolate(method=interp_method.lower(), limit_direction='both')
    if integer:
        return key_frame_series.astype(int)
    return key_frame_series




if key_frames:
    try:
        angle_series = get_inbetweens(parse_key_frames(angle))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `angle` correctly for key frames.\n"
            "Attempting to interpret `angle` as "
            f'"0: ({angle})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        angle = f"0: ({angle})"
        angle_series = get_inbetweens(parse_key_frames(angle))

    try:
        zoom_series = get_inbetweens(parse_key_frames(zoom))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `zoom` correctly for key frames.\n"
            "Attempting to interpret `zoom` as "
            f'"0: ({zoom})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        zoom = f"0: ({zoom})"
        zoom_series = get_inbetweens(parse_key_frames(zoom))

    try:
        translation_x_series = get_inbetweens(parse_key_frames(translation_x))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `translation_x` correctly for key frames.\n"
            "Attempting to interpret `translation_x` as "
            f'"0: ({translation_x})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        translation_x = f"0: ({translation_x})"
        translation_x_series = get_inbetweens(parse_key_frames(translation_x))

    try:
        translation_y_series = get_inbetweens(parse_key_frames(translation_y))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `translation_y` correctly for key frames.\n"
            "Attempting to interpret `translation_y` as "
            f'"0: ({translation_y})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        translation_y = f"0: ({translation_y})"
        translation_y_series = get_inbetweens(parse_key_frames(translation_y))

else:
    angle = float(angle)
    zoom = float(zoom)
    translation_x = float(translation_x)
    translation_y = float(translation_y)

# 3.3: EXTRA SETTINGS


# @markdown ####**Saving:**

intermediate_saves = [x for x in range(steps)]  # @param{type: 'raw'}
intermediates_in_subfolder = True  # @param{type: 'boolean'}
# @markdown Intermediate steps will save a copy at your specified intervals. You can either format it as a single integer or a list of specific steps 

# @markdown A value of `2` will save a copy at 33% and 66%. 0 will save none.

# @markdown A value of `[5, 9, 34, 45]` will save at steps 5, 9, 34, and 45. (Make sure to include the brackets)


if type(intermediate_saves) is not list:
    if intermediate_saves:
        steps_per_checkpoint = math.floor((steps - skip_steps - 1) // (intermediate_saves + 1))
        steps_per_checkpoint = steps_per_checkpoint if steps_per_checkpoint > 0 else 1
        print(f'Will save every {steps_per_checkpoint} steps')
    else:
        steps_per_checkpoint = steps + 10
else:
    steps_per_checkpoint = None



    # @markdown ---

# @markdown ####**SuperRes Sharpening:**
# @markdown *Sharpen each image using latent-diffusion. Does not run in animation mode. `keep_unsharp` will save both versions.*
sharpen_preset = 'Off'  # @param ['Off', 'Faster', 'Fast', 'Slow', 'Very Slow']
keep_unsharp = True  # @param{type: 'boolean'}



    # @markdown ---

# @markdown ####**Advanced Settings:**
# @markdown *There are a few extra advanced settings available if you double click this cell.*

# @markdown *Perlin init will replace your init, so uncheck if using one.*

perlin_init = False  # @param{type: 'boolean'}
perlin_mode = 'mixed'  # @param ['mixed', 'color', 'gray']
set_seed = 'random_seed'  # @param{type: 'string'}
eta = 0.8  # @param{type: 'number'}
clamp_grad = True  # @param{type: 'boolean'}
clamp_max = 0.05  # @param{type: 'number'}

### EXTRA ADVANCED SETTINGS:
randomize_class = True
clip_denoised = False
fuzzy_prompt = False
rand_mag = 0.05

# @markdown ---

# @markdown ####**Cutn Scheduling:**
# @markdown Format: `[40]*400+[20]*600` = 40 cuts for the first 400 /1000 steps, then 20 for the last 600/1000

# @markdown cut_overview and cut_innercut are cumulative for total cutn on any given step. Overview cuts see the entire image and are good for early structure, innercuts are your standard cutn.

cut_overview = "[12]*400+[4]*600"  # @param {type: 'string'}       
cut_innercut = "[4]*400+[12]*600"  # @param {type: 'string'}  
cut_ic_pow = 1  # @param {type: 'number'}  
cut_icgray_p = "[0.2]*400+[0]*600"  # @param {type: 'string'}  

# 3.4: PROMPTS

# CHANGE THE PROMPTS
text_prompts = {
    "0": ["I still pretend you're only out of sight in another room, smiling at your phone...:3",
          "Melancholy, grieving the loss of a loved one, reminiscing on the happy times with them:1"]
}

image_prompts = {
    # 0:['content/init_images/colored question mark.png:2',],
}

# 4: DO THE RUN
# @title Do the Run!
# @markdown `n_batches` ignored with animation modes.
display_rate = 1  # @param{type: 'number'}
n_batches = 1  # @param{type: 'number'}

batch_size = 1


def move_files(start_num, end_num, old_folder, new_folder):
    for i in range(start_num, end_num):
        old_file = old_folder + f'/{batch_name}({batchNum})_{i:04}.png'
        new_file = new_folder + f'/{batch_name}({batchNum})_{i:04}.png'
        os.rename(old_file, new_file)


# @markdown ---


resume_run = False  # @param{type: 'boolean'}
run_to_resume = 'latest'  # @param{type: 'string'}
resume_from_frame = 'latest'  # @param{type: 'string'}
retain_overwritten_frames = False  # @param{type: 'boolean'}



skip_step_ratio = int(frames_skip_steps.rstrip("%")) / 100
calc_frames_skip_steps = math.floor(steps * skip_step_ratio)

if steps <= calc_frames_skip_steps:
    sys.exit("ERROR: You can't skip more steps than your total steps")

if set_seed == 'random_seed':
    random.seed()
    seed = random.randint(0, 2 ** 32)
    # print(f'Using seed: {seed}')
else:
    seed = int(set_seed)

# ====================================
#            RUN DIFFUSION
# ====================================
# Below, we'll run diffusion across each of the different argument dicts we've parsed
for cur_arg_idx, cur_arg_dict in enumerate(arg_dicts):

    # Creating a folder for this current batch 
    batchFolder = f'{outDirPath}/{cur_arg_dict["batch_name"]}'
    createPath(batchFolder)

    # Save the config file in the resulting folder
    with open(f"{batchFolder}/{cur_arg_dict['batch_name']}_config.json", "w") as config_json:
        json.dump(unprocessed_arg_dicts[cur_arg_idx], config_json, indent=2)

    # Delete the original config file
    os.remove(str(config_paths[cur_arg_idx]))

    args = SimpleNamespace(**cur_arg_dict)

    if args.intermediate_saves and args.intermediates_in_subfolder is True:
        partialFolder = f'{batchFolder}/partials'
        createPath(partialFolder)

    if args.sharpen_preset != 'Off' and args.keep_unsharp is True:
        unsharpenFolder = f'{batchFolder}/unsharpened'
        createPath(unsharpenFolder)

    if retain_overwritten_frames is True:
        retainFolder = f'{batchFolder}/retained'
        createPath(retainFolder)
    
    if diffusion_model == '512x512_diffusion_uncond_finetune_008100':
        model_config.update({
            'attention_resolutions': '32, 16, 8',
            'class_cond': False,
            'diffusion_steps': args.diffusion_steps,
            'rescale_timesteps': True,
            'timestep_respacing': args.timestep_respacing,
            'image_size': 512,
            'learn_sigma': True,
            'noise_schedule': 'linear',
            'num_channels': 256,
            'num_head_channels': 64,
            'num_res_blocks': 2,
            'resblock_updown': True,
            'use_checkpoint': use_checkpoint,
            'use_fp16': True,
            'use_scale_shift_norm': True,
        })

    elif diffusion_model == '256x256_diffusion_uncond':
        model_config.update({
            'attention_resolutions': '32, 16, 8',
            'class_cond': False,
            'diffusion_steps': args.diffusion_steps,
            'rescale_timesteps': True,
            'timestep_respacing': args.timestep_respacing,
            'image_size': 256,
            'learn_sigma': True,
            'noise_schedule': 'linear',
            'num_channels': 256,
            'num_head_channels': 64,
            'num_res_blocks': 2,
            'resblock_updown': True,
            'use_checkpoint': use_checkpoint,
            'use_fp16': True,
            'use_scale_shift_norm': True,
        })

    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(torch.load(f'{model_path}/{diffusion_model}.pt', map_location='cpu'))
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if 'qkv' in name or 'norm' in name or 'proj' in name:
            param.requires_grad_()
    if model_config['use_fp16']:
        model.convert_to_fp16()

    gc.collect()
    torch.cuda.empty_cache()
    try:
        do_run(args)
    except KeyboardInterrupt:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    sleep(10)
