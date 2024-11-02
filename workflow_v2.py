import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import time
import numpy as np
from PIL import Image
import gc
import json

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
# add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()
        
from nodes import (
    UNETLoader,
    DualCLIPLoader,
    VAEDecode,
    VAELoader,
    LoadImage,
    ImageScale,
    CLIPTextEncode,
    VAEEncode,
    NODE_CLASS_MAPPINGS,
)


class Interior:
    def __init__(self):
        import_custom_nodes()
        with torch.inference_mode():
            self.loadimage = LoadImage()
            self.imagescale = ImageScale()
            self.dualcliploader = DualCLIPLoader()
            self.vaeencode = VAEEncode()
            self.efficient_loader = NODE_CLASS_MAPPINGS["Efficient Loader"]()
            # self.showtextpysssss = NODE_CLASS_MAPPINGS["ShowText|pysssss"]()
            self.text_concatenate = NODE_CLASS_MAPPINGS["Text Concatenate"]()
            # self.textboxfofo = NODE_CLASS_MAPPINGS["TextBox|fofo"]()
            self.controlnetloaderadvanced = NODE_CLASS_MAPPINGS["ControlNetLoaderAdvanced"]()
            self.cliptextencode = CLIPTextEncode()
            self.ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
            self.clipattentionmultiply = NODE_CLASS_MAPPINGS["CLIPAttentionMultiply"]()
            self.randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
            self.ipadapterstylecomposition = NODE_CLASS_MAPPINGS["IPAdapterStyleComposition"]()
            self.ipadapterunifiedloader = NODE_CLASS_MAPPINGS["IPAdapterUnifiedLoader"]()
            self.maskfromcolor = NODE_CLASS_MAPPINGS["MaskFromColor+"]()
            self.imagegaussianblur = NODE_CLASS_MAPPINGS["ImageGaussianBlur"]()
            self.acn_advancedcontrolnetapply = NODE_CLASS_MAPPINGS["ACN_AdvancedControlNetApply"]()
            self.anylineartpreprocessor_aux = NODE_CLASS_MAPPINGS["AnyLineArtPreprocessor_aux"]()
            self.vaeloader = VAELoader()
            self.ksampler_efficient = NODE_CLASS_MAPPINGS["KSampler (Efficient)"]()
            self.fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
            self.basicguider = NODE_CLASS_MAPPINGS["BasicGuider"]()
            self.basicscheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]()
            self.samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
            self.vaedecode = VAEDecode()
            self.image_save = NODE_CLASS_MAPPINGS["Image Save"]()
            self.easy_cleangpuused = NODE_CLASS_MAPPINGS["easy cleanGpuUsed"]()
            self.imagescaletototalpixels = NODE_CLASS_MAPPINGS["ImageScaleToTotalPixels"]()
            self.scaledsoftcontrolnetweights = NODE_CLASS_MAPPINGS["ScaledSoftControlNetWeights"]()
            self.unetloader = UNETLoader()
            
            self.controlnetloaderadvanced_227 = self.controlnetloaderadvanced.load_controlnet(
                control_net_name="sdxl_depth.safetensors"
            )
            self.controlnetloaderadvanced_1457 = self.controlnetloaderadvanced.load_controlnet(
                control_net_name="mistoLine_rank256.safetensors"
            )
            self.vaeloader_2162 = self.vaeloader.load_vae(vae_name="ae.safetensors")
            self.unetloader_2164 = self.unetloader.load_unet(unet_name="flux1-dev-fp8.safetensors", weight_dtype="fp8_e4m3fn")
            self.dualcliploader_2161 = self.dualcliploader.load_clip(
                clip_name1="t5xxl_fp16.safetensors",
                clip_name2="clip_l.safetensors",
                type="flux",
            )
            with open('styles.json', 'r') as file:
                self.style_json = json.load(file)
    @torch.inference_mode()
    def __call__(self, *args, **kwargs):
        st = time.time()
        user_prompt = kwargs.get('prompt')
        style = kwargs.get('style')
        print("style: ", style)
        style_img = random.choice(os.listdir(f"ComfyUI/input/Styles/{style}"))
        style_img = f"Styles/{style}/{style_img.split('/')[-1]}"
        print("style_img: ", style_img)
        creativity = kwargs.get('creativity')
        denoise = round(0.3 + (0.65 - 0.3)*(int(creativity)/100),2)
        print("denoise: ",denoise)
        # Load images
        loadimage_1633 = self.loadimage.load_image(image="door.png")
        loadimage_1631 = self.loadimage.load_image(image="window.png")
        loadimage_1637 = self.loadimage.load_image(image="mask_image.png")
        loadimage_2155 = self.loadimage.load_image(image=style_img)
        loadimage_326 = self.loadimage.load_image(image="depth_image.png")
        loadimage_554 = self.loadimage.load_image(image="input_image.png")

        # Scale images
        imagescale_1645 = self.imagescale.upscale(
            upscale_method="lanczos",
            width=1920,
            height=0,
            crop="disabled",
            image=get_value_at_index(loadimage_554, 0),
        )

        imagescale_2129 = self.imagescale.upscale(
            upscale_method="lanczos",
            width=1920,
            height=0,
            crop="disabled",
            image=get_value_at_index(loadimage_1637, 0),
        )

        # Text processing
        #style_prompt = f"{user_prompt} {self.style_json[style]} bright interior"
        style_prompt = f"{self.style_json[style]}"
        final_prompt = f"8k interior photogrpahy, high quality image of {style_prompt}(({user_prompt}):1.2)"
        print("final_prompt: ", final_prompt)
        # Model loading and processing
        efficient_loader_218 = self.efficient_loader.efficientloader(
            ckpt_name="hyperray_SDXL.safetensors",
            vae_name="Baked VAE",
            clip_skip=-1,
            lora_name="None",
            lora_model_strength=1,
            lora_clip_strength=1,
            positive=final_prompt,
            negative="lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, disfigured, username, watermark, signature, backgound out of focus, black and white, monochrome",
            token_normalization="none",
            weight_interpretation="comfy",
            empty_latent_width=1280,
            empty_latent_height=704,
            batch_size=1,
        )
       progress_25 = {
                    "data": {
                        "progress":25
                    }
                    }
        print("json_data: ", progress_25)
        resp_10 = put_orders(order_id, progress_25, token)
        print("response 25: ", resp_25.text)
        ipadapterunifiedloader_1105 = self.ipadapterunifiedloader.load_models(
            preset="STANDARD (medium strength)",
            model=get_value_at_index(efficient_loader_218, 0),
        )

        vaeencode_1100 = self.vaeencode.encode(
            pixels=get_value_at_index(imagescale_1645, 0),
            vae=get_value_at_index(efficient_loader_218, 4),
        )

        ksamplerselect_1156 = self.ksamplerselect.get_sampler(sampler_name="euler")
        randomnoise_1159 = self.randomnoise.get_noise(noise_seed=random.randint(1, 2**64))

        # IPAdapter processing
        ipadapterstylecomposition_559 = self.ipadapterstylecomposition.apply_ipadapter(
            weight_style=0.75,
            weight_composition=0,
            expand_style=False,
            combine_embeds="add",
            start_at=0,
            end_at=1,
            embeds_scaling="V only",
            model=get_value_at_index(ipadapterunifiedloader_1105, 0),
            ipadapter=get_value_at_index(ipadapterunifiedloader_1105, 1),
            image_style=get_value_at_index(loadimage_2155, 0),
            image_composition=get_value_at_index(loadimage_2155, 0),
        )

        maskfromcolor_1628 = self.maskfromcolor.execute(
            red=255,
            green=0,
            blue=0,
            threshold=50,
            image=get_value_at_index(imagescale_2129, 0),
        )

        ipadapterstylecomposition_1630 = self.ipadapterstylecomposition.apply_ipadapter(
            weight_style=0.7000000000000001,
            weight_composition=0.7000000000000001,
            expand_style=False,
            combine_embeds="add",
            start_at=0,
            end_at=1,
            embeds_scaling="V only",
            model=get_value_at_index(ipadapterstylecomposition_559, 0),
            ipadapter=get_value_at_index(ipadapterunifiedloader_1105, 1),
            image_style=get_value_at_index(loadimage_1631, 0),
            image_composition=get_value_at_index(loadimage_1631, 0),
            attn_mask=get_value_at_index(maskfromcolor_1628, 0),
        )

        maskfromcolor_1635 = self.maskfromcolor.execute(
            red=0,
            green=0,
            blue=255,
            threshold=50,
            image=get_value_at_index(imagescale_2129, 0),
        )

        ipadapterstylecomposition_1632 = self.ipadapterstylecomposition.apply_ipadapter(
            weight_style=0.7000000000000001,
            weight_composition=0.7000000000000001,
            expand_style=False,
            combine_embeds="add",
            start_at=0,
            end_at=1,
            embeds_scaling="V only",
            model=get_value_at_index(ipadapterstylecomposition_1630, 0),
            ipadapter=get_value_at_index(ipadapterunifiedloader_1105, 1),
            image_style=get_value_at_index(loadimage_1633, 0),
            image_composition=get_value_at_index(loadimage_1633, 0),
            attn_mask=get_value_at_index(maskfromcolor_1635, 0),
        )

        # Image processing and ControlNet
        imagegaussianblur_1276 = self.imagegaussianblur.image_gaussian_blur(
            radius=2, images=get_value_at_index(loadimage_326, 0)
        )

        acn_advancedcontrolnetapply_222 = self.acn_advancedcontrolnetapply.apply_controlnet(
            strength=0.9500000000000001,
            start_percent=0,
            end_percent=0.84,
            positive=get_value_at_index(efficient_loader_218, 1),
            negative=get_value_at_index(efficient_loader_218, 2),
            control_net=get_value_at_index(self.controlnetloaderadvanced_227, 0),
            image=get_value_at_index(imagegaussianblur_1276, 0),
        )

        anylineartpreprocessor_aux_1454 = self.anylineartpreprocessor_aux.get_anyline(
            merge_with_lineart="lineart_realisitic",
            resolution=1920,
            lineart_lower_bound=0,
            lineart_upper_bound=1,
            object_min_size=36,
            object_connectivity=1,
            image=get_value_at_index(loadimage_554, 0),
        )

        acn_advancedcontrolnetapply_1456 = self.acn_advancedcontrolnetapply.apply_controlnet(
            strength=0.65,
            start_percent=0,
            end_percent=0.84,
            positive=get_value_at_index(acn_advancedcontrolnetapply_222, 0),
            negative=get_value_at_index(acn_advancedcontrolnetapply_222, 1),
            control_net=get_value_at_index(self.controlnetloaderadvanced_1457, 0),
            image=get_value_at_index(anylineartpreprocessor_aux_1454, 0),
        )

        # Sampling and decoding
        ksampler_efficient_214 = self.ksampler_efficient.sample(
            seed=random.randint(1, 2**64),
            steps=15,
            cfg=1,
            sampler_name="euler_ancestral",
            scheduler="normal",
            denoise=1,
            preview_method="auto",
            vae_decode="true",
            model=get_value_at_index(ipadapterstylecomposition_1632, 0),
            positive=get_value_at_index(acn_advancedcontrolnetapply_1456, 0),
            negative=get_value_at_index(acn_advancedcontrolnetapply_1456, 1),
            latent_image=get_value_at_index(vaeencode_1100, 0),
            optional_vae=get_value_at_index(efficient_loader_218, 4),
        )

       progress_75 = {
                    "data": {
                        "progress":75
                    }
                    }
        print("json_data: ", progress_75)
        resp_10 = put_orders(order_id, progress_75, token)
        print("response 75: ", resp_75.text)

        easy_cleangpuused_1492 = self.easy_cleangpuused.empty_cache(
            anything=get_value_at_index(ksampler_efficient_214, 5),
            unique_id=9358060924967246150,
        )
        # del ksampler_efficient_214
        

        vaeencode_1173 = self.vaeencode.encode(
            pixels=get_value_at_index(ksampler_efficient_214, 5),
            vae=get_value_at_index(self.vaeloader_2162, 0),
        )

        scaledsoftcontrolnetweights_1691 = self.scaledsoftcontrolnetweights.load_weights(
            base_multiplier=0.84, flip_weights=False, uncond_multiplier=1
        )

        clipattentionmultiply_2163 = self.clipattentionmultiply.patch(
            q=1.2,
            k=1.1,
            v=0.8,
            out=1.25,
            clip=get_value_at_index(self.dualcliploader_2161, 0),
        )

        cliptextencode_1152 = self.cliptextencode.encode(
            text=final_prompt,
            clip=get_value_at_index(clipattentionmultiply_2163, 0),
        )
        
        fluxguidance_1160 = self.fluxguidance.append(
            guidance=3.5, conditioning=get_value_at_index(cliptextencode_1152, 0)
        )

        basicguider_1158 = self.basicguider.get_guider(
            model=get_value_at_index(self.unetloader_2164, 0),
            conditioning=get_value_at_index(fluxguidance_1160, 0),
        )

        basicscheduler_1157 = self.basicscheduler.get_sigmas(
            scheduler="normal",
            steps=45,
            denoise=denoise,
            model=get_value_at_index(self.unetloader_2164, 0),
        )

        samplercustomadvanced_1155 = self.samplercustomadvanced.sample(
            noise=get_value_at_index(randomnoise_1159, 0),
            guider=get_value_at_index(basicguider_1158, 0),
            sampler=get_value_at_index(ksamplerselect_1156, 0),
            sigmas=get_value_at_index(basicscheduler_1157, 0),
            latent_image=get_value_at_index(vaeencode_1173, 0),
        )

        vaedecode_1153 = self.vaedecode.decode(
            samples=get_value_at_index(samplercustomadvanced_1155, 0),
            vae=get_value_at_index(self.vaeloader_2162, 0),
        )

        easy_cleangpuused_1486 = self.easy_cleangpuused.empty_cache(
            anything=get_value_at_index(vaedecode_1153, 0),
            unique_id=3713307099911164865,
        )

        # Final image processing
        imgs = []
        for res in vaedecode_1153[0]:
            img = Image.fromarray(np.clip(255. * res.detach().cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
            imgs.append(img)
        
        et = time.time()
        print('infer time: ', et-st)
        return imgs
