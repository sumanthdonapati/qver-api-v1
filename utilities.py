import requests
import secrets
from urllib.request import urlopen
from flask import jsonify, make_response, send_file
import os
import io
from io import BytesIO
import base64
from typing import Union
import PIL
import PIL.ImageOps
from PIL import Image
    
gpu_metrics = {}

def get_seed(seed:int = -1) -> int:
  if seed == -1:
      seed = secrets.randbelow(99999999999999)
  return int(seed)

def get_error_response(msg,code,type="server"):
    if type=="server":
        response = make_response(jsonify(error= msg),code)
        response.headers['x-gpu-info'] = gpu_metrics
    if type=="serverless":
        response = [{"error":msg,"status_code":code},{"x-gpu-info":gpu_metrics}]
    return response
    
def load_image(image: Union[str, PIL.Image.Image]) -> PIL.Image.Image:
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
    Returns:
        `PIL.Image.Image`:
            A PIL Image.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = PIL.Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            try:
                im_binary = base64.b64decode(image)
                buf = io.BytesIO(im_binary)
            except:
                missing_padding = len(image) % 4
                image += '=' * (4 - missing_padding)
                im_binary = base64.b64decode(image)
                buf = io.BytesIO(im_binary)
                
            image = Image.open(buf)
            
    image = PIL.ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

def resize_image(image_path, max_dimension=2048):
    img = image_path
    width, height = img.size
    
    if width > max_dimension or height > max_dimension:
        # Calculate the aspect ratio
        aspect_ratio = width / height

        if width > height:
            new_width = max_dimension
            new_height = int(max_dimension / aspect_ratio)
        else:
            new_height = max_dimension
            new_width = int(max_dimension * aspect_ratio)
        try:       
            img = img.resize((new_width, new_height), Image.ANTIALIAS)
        except:
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return img

def upload_image(image_path,bearer_token):
    files = {'files': open(image_path, 'rb')}
    url = 'https://strapi.qver.ai/api/upload'
    headers = {'Authorization': f'Bearer {bearer_token}'}
    response = requests.post(url, files=files, headers=headers)
    print('upload status: ',response.status_code)
    for file in files.values():
        file.close()
    img_url = response.json()[0]['url']
    return img_url
    
def orders(json_file,bearer_token):
    url = 'https://strapi.qver.ai/api/orders'
    headers = {'Authorization': f'Bearer {bearer_token}'}
    response = requests.post(url, json=json_file, headers=headers)
    print('orders status: ',response.status_code)
    if response.status_code != 200:
        print("order response: ", response.text)
    return response

def check_order(id,bearer_token):
    url = f'https://strapi.qver.ai/api/orders/{id}'
    headers = {'Authorization': f'Bearer {bearer_token}'}
    response = requests.get(url, headers=headers)
    print('check order status: ',response.status_code)
    if response.status_code != 200:
        print("check order response: ", response.text)
    json_data = response.json()
    return json_data["data"]

def put_orders(id, json_file, bearer_token):
    api_url = f"https://strapi.qver.ai/api/orders/{id}"
    headers = {'Authorization': f'Bearer {bearer_token}'}
    response = requests.put(api_url, json=json_file, headers=headers)
    print('put orders status:', response.status_code)
    if response.status_code != 200:
        print("put order response:", response.text)
    return response
