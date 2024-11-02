import json
import os
from PIL import Image
from glob import glob
import datetime
import numpy as np
import time
import random
import traceback

import sys
import os
from utilities import *
from workflow_v2 import *
import runpod
## GPU metadata
import httpcore
setattr(httpcore, 'SyncHTTPTransport', 'AsyncHTTPProxy')
from googletrans import Translator
translator = Translator()

##Flask app
generator = Interior()
# imgs = generator()
all_styles = ["artDeco","avantGarde","boho","classic","contemporary","eco","fusion","highTech","industrial","loft","memphis","midCenturyModern","minimalism","modern","nautical","neoclassicism","provence","rustic","scandinavian","victorian"]        

def interior_infer(job):
    start = time.time()            
    jsonFile = job["input"] #request.get_json()
    try:
    # if True:
        body = jsonFile
        print("body: ",body)
        order_id = body.get("id")
        prompt = body.get("prompt")
        input_image = body.get("object_image")
        mask_image = body.get("mask_image")
        depth_image = body.get("depth_map_image")
        style = body.get("style")
        token = body.get('token')
        creativity = body.get("creativity")
        base_64 = body.get("base64",False)

        try:
            data = check_order(order_id,token)
            print("data: ",data)
            if data['attributes']['final_image'] is not None:
                return get_error_response("request already processed",400,type="serverless")
        except:
            print('order validation failed')
            pass

        if input_image is None:
            return get_error_response("image not found",400,type="serverless")
        else:
            try:
                input_img = load_image(input_image)
                input_img.save("ComfyUI/input/input_image.png")
            except:
                print("image trace back:",str(traceback.format_exc()))
                return get_error_response("Invalid Image",400,type="serverless")

        if depth_image is None:
            return get_error_response("image not found",400,type="serverless")
        else:
            try:
                depth_img = load_image(depth_image)
                depth_img.save("ComfyUI/input/depth_image.png")
            except:
                print("image trace back:",str(traceback.format_exc()))
                return get_error_response("Invalid Image",400,type="serverless")
        
        if mask_image is None:
            return get_error_response("image not found",400,type="serverless")
        else:
            try:
                mask_img = load_image(mask_image)
                mask_img.save("ComfyUI/input/mask_image.png")
            except:
                print("image trace back:",str(traceback.format_exc()))
                return get_error_response("Invalid Image",400,type="serverless")

        if style not in all_styles:
            return get_error_response(f"{style} is invalid Style",400,type="serverless")
        try: 
            translation = translator.translate(prompt, dest='en')
            prompt = translation.text
        except Exception as e:
            print("trans error ignored: ",e)
        print(prompt)
        progress_10 = {
                    "data": {
                        "progress":10
                    }
                    }
        print("json_data: ", progress_10)
        resp_10 = put_orders(order_id, progress_10, token)
        print("response: ", resp_10.text)
        infer_start=time.time()
        imgs = generator(prompt=prompt,style=style,creativity=creativity, order_id=order_id, token=token) 
        infer_end=time.time()
        infer_time = [infer_end-infer_start]
        
        imgs[0].save('outputs/final_image.png')
        
        img_url = upload_image('outputs/final_image.png',token)
        final_image = "https://strapi.qver.ai"+img_url
        json_data = {
                    "data": {
                        "object_image": input_image,
                        "depth_map_image": depth_image,
                        "mask_image": mask_image,
                        "final_image": final_image,
                        "prompt": prompt,
                        "style": style,
                        "progress":100
                    }
                    }
        print("json_data: ", json_data)
        resp = put_orders(order_id, json_data, token)
        print("response: ", resp.text)
        # print(f"JSON Body:{body['scale']}")
        print(f'API Request start time:', start)
        print(f'Pre-processing Time:{infer_start-start}')
        print(f'Generation Time:{infer_time[0]}')

        response = [{"status":"Sucess"}]
        
        print(f'Post-processing Time:{time.time()-infer_end}')
        print(f'API Reponse Time:{time.time()}')
        
        return response
        
    except Exception as e:
        print(f'Error:{e}')
        print("trace back:",str(traceback.format_exc()))
        return get_error_response("Internal Server Error!",500,type="serverless")

print("running serverless")
runpod.serverless.start({"handler": interior_infer})
