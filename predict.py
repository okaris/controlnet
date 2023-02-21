# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/resolve/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
from subprocess import call
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler
from PIL import Image
import numpy as np
from typing import List
import zipfile
from io import BytesIO
import urllib.request
import shutil
import subprocess
from utils import get_state_dict_path, download_model, model_dl_urls, annotator_dl_urls
import torch
# from share import *
from cldm.model import load_state_dict

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        return True

    def predict(
        self,
        controlnet_model: str = Input(
            description="Type of ControlNet model to use",
            choices=["Canny", "Depth", "HED", "Normal", "MLSD", "OpenPose", "Scribble", "Seg"],
            default=None,
        ),
        base_model: str = Input(
            description="Type of base model to use",
            default=None
        ),
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Prompt for the model"),
        num_samples: str = Input(
            description="Number of samples (higher values may OOM)",
            choices=['1', '4'],
            default='1'
        ),
        image_resolution: str = Input(
            description="Image resolution to be generated",
            choices = ['256', '512', '768'],
            default='512'
        ),
        low_threshold: int = Input(description="Canny low threshold (only applicable when model type is 'canny')", default=100, ge=1, le=255), # only applicable when model type is 'canny'
        high_threshold: int = Input(description="Canny high threshold (only applicable when model type is 'canny')", default=200, ge=1, le=255), # only applicable when model type is 'canny'
        ddim_steps: int = Input(description="Steps", default=20),
        scale: float = Input(description="Guidance Scale", default=9.0, ge=0.1, le=30.0),
        seed: int = Input(description="Seed", default=None),
        eta: float = Input(description="eta (DDIM)", default=0.0),
        a_prompt: str = Input(description="Added Prompt", default="best quality, extremely detailed"),
        n_prompt: str = Input(description="Negative Prompt", default="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"),
        detect_resolution: int = Input(description="Resolution for detection)", default=512, ge=128, le=1024), # only applicable when model type is 'HED', 'seg', or 'MLSD'
        bg_threshold: float = Input(description="Background Threshold (only applicable when model type is 'normal')", default=0.0, ge=0.0, le=1.0), # only applicable when model type is 'normal'
        value_threshold: float = Input(description="Value Threshold (only applicable when model type is 'MLSD')", default=0.1, ge=0.01, le=2.0), # only applicable when model type is 'MLSD'
        distance_threshold: float = Input(description="Distance Threshold (only applicable when model type is 'MLSD')", default=0.1, ge=0.01, le=20.0), # only applicable when model type is 'MLSD'
    ) -> List[Path]:
        """Run a single prediction on the model"""
        num_samples = int(num_samples)
        image_resolution = int(image_resolution)
        if not seed:
            seed = np.random.randint(1000000)
        else:
            seed = int(seed)

        # load input_image
        input_image = Image.open(image)
        # convert to numpy
        input_image = np.array(input_image)
        

        def download_ckpt(self, url):
            # Download the file to models dir
            print("Downloading model...")
            filename = url.split("/")[-1]
            filepath = os.path.join("models", filename)
            #if file doesn't exist, download it
            if not os.path.exists(filepath):
                urllib.request.urlretrieve(url, filepath)
                print("Downloaded model to {}".format(filepath))
            else:
                print("Model already exists at {}".format(filepath))

        case = controlnet_model.lower()

        download_ckpt(self, "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_{}.pth".format(case))
        self.controlnet_model = 'control_sd15_{}.pth'.format(case)

        if base_model is not None:
            download_ckpt(self, base_model)
            download_ckpt(self, "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt")
            path_sd15 = './models/v1-5-pruned.ckpt'
            path_sd15_with_control = './models/control_sd15_{}.pth'.format(case)
            path_input = './models/{}.ckpt'.format(base_model.split("/")[-1].split(".")[0])
            path_output = './models/control_sd15_{}_{}.pth'.format(case, base_model)

            assert os.path.exists(path_sd15), 'Input path_sd15 does not exists!'
            assert os.path.exists(path_sd15_with_control), 'Input path_sd15_with_control does not exists!'
            assert os.path.exists(path_input), 'Input path_input does not exists!'
            assert os.path.exists(os.path.dirname(path_output)), 'Output folder not exists!'

            sd15_state_dict = load_state_dict(path_sd15)
            sd15_with_control_state_dict = load_state_dict(path_sd15_with_control)
            input_state_dict = load_state_dict(path_input)
            
            def get_node_name(name, parent_name):
                if len(name) <= len(parent_name):
                    return False, ''
                p = name[:len(parent_name)]
                if p != parent_name:
                    return False, ''
                return True, name[len(parent_name):]


            keys = sd15_with_control_state_dict.keys()

            final_state_dict = {}
            for key in keys:
                is_first_stage, _ = get_node_name(key, 'first_stage_model')
                is_cond_stage, _ = get_node_name(key, 'cond_stage_model')
                if is_first_stage or is_cond_stage:
                    final_state_dict[key] = input_state_dict[key]
                    continue
                p = sd15_with_control_state_dict[key]
                is_control, node_name = get_node_name(key, 'control_')
                if is_control:
                    sd15_key_name = 'model.diffusion_' + node_name
                else:
                    sd15_key_name = key
                if sd15_key_name in input_state_dict:
                    p_new = p + input_state_dict[sd15_key_name] - sd15_state_dict[sd15_key_name]
                    # print(f'Offset clone from [{sd15_key_name}] to [{key}]')
                else:
                    p_new = p
                    # print(f'Direct clone to [{key}]')
                final_state_dict[key] = p_new

            torch.save(final_state_dict, path_output)
            self.controlnet_model = 'control_sd15_{}_{}.pth'.format(case, base_model)
            print('Transferred model saved at ' + path_output)


        self.model = create_model('./models/cldm_v15.yaml').cuda()
        self.model.load_state_dict(load_state_dict(get_state_dict_path(self.controlnet_model), location='cuda'))
        self.ddim_sampler = DDIMSampler(self.model)

        if case == "canny":
            from gradio_canny2image import process_canny
            outputs = process_canny(
                input_image,
                prompt,
                a_prompt,
                n_prompt,
                num_samples,
                image_resolution,
                ddim_steps,
                scale,
                seed,
                eta,
                low_threshold,
                high_threshold,
                self.model,
                self.ddim_sampler,
            )
        elif case == "depth":
            from gradio_depth2image import process_depth
            outputs = process_depth(
                input_image,
                prompt,
                a_prompt,
                n_prompt,
                num_samples,
                image_resolution,
                detect_resolution,
                ddim_steps,
                scale,
                seed,
                eta,
                self.model,
                self.ddim_sampler,
            )
        elif case == "hed":
            from gradio_hed2image import process_hed
            outputs = process_hed(
                input_image,
                prompt,
                a_prompt,
                n_prompt,
                num_samples,
                image_resolution,
                detect_resolution,
                ddim_steps,
                scale,
                seed,
                eta,
                self.model,
                self.ddim_sampler,
            )
        elif case == "normal":
            from gradio_normal2image import process_normal
            outputs = process_normal(
                input_image,
                prompt,
                a_prompt,
                n_prompt,
                num_samples,
                image_resolution,
                ddim_steps,
                scale,
                seed,
                eta,
                bg_threshold,
                self.model,
                self.ddim_sampler,
            )
        elif case == "mlsd":
            from gradio_hough2image import process_mlsd
            outputs = process_mlsd(
                input_image,
                prompt,
                a_prompt,
                n_prompt,
                num_samples,
                image_resolution,
                detect_resolution,
                ddim_steps,
                scale,
                seed,
                eta,
                value_threshold,
                distance_threshold,
                self.model,
                self.ddim_sampler,
            )
        elif case == "scribble":
            from gradio_scribble2image import process_scribble
            outputs = process_scribble(
                input_image,
                prompt,
                a_prompt,
                n_prompt,
                num_samples,
                image_resolution,
                ddim_steps,
                scale,
                seed,
                eta,
                self.model,
                self.ddim_sampler,
            )
        elif case == "seg":
            from gradio_seg2image import process_seg
            outputs = process_seg(
                input_image,
                prompt,
                a_prompt,
                n_prompt,
                num_samples,
                image_resolution,
                detect_resolution,
                ddim_steps,
                scale,
                seed,
                eta,
                self.model,
                self.ddim_sampler,
            )
        else:
            raise ValueError("Invalid controlnet_model: {}".format(controlnet_model))
            
        # outputs from list to PIL
        outputs = [Image.fromarray(output) for output in outputs]
        # save outputs to file
        outputs = [output.save(f"tmp/output_{i}.png") for i, output in enumerate(outputs)]
        # return paths to output files
        return [Path(f"tmp/output_{i}.png") for i in range(len(outputs))]
