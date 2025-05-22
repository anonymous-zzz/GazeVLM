from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import math

import json
import csv
import time

##########################################################################
##########################################################################
##########################################################################
# model_name = "llava-hf/llama3-llava-next-8b-hf"
model_name = "llava-hf/llava-v1.6-mistral-7b-hf" 
#------------------------------------------------------------------------#
image_type = 2      # (0) Raw
                    # (1) Preprocessing-only
                    # (2) Preprocessing + Postprocessing
token_total = 200   # total token budget
crop_alpha = 0.5    # (1.0) two global-view and local-view images with the same size
gaze_beta = 0.5     # (1.0) closest, (0.0) uniform
#------------------------------------------------------------------------#
if image_type == 0:
    image_preprocessing_type = False
    image_postprocessing_type = False
elif image_type == 1:
    image_preprocessing_type = True
    image_postprocessing_type = False
elif image_type == 2:
    image_preprocessing_type = True
    image_postprocessing_type = True
#------------------------------------------------------------------------#
if image_preprocessing_type:
    selected_crop_size = [int((672//2) * crop_alpha), \
                    (672//2)] 
force_resolution = [672, 672]
##########################################################################
##########################################################################
##########################################################################

processor = LlavaNextProcessor.from_pretrained(model_name)
model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.to("cuda:0")

model.config.foveated_config = { }
model.config.foveated_config['force_resolution'] = force_resolution
model.config.foveated_config['image_preprocessing_type'] = image_preprocessing_type
model.config.foveated_config['image_postprocessing_type'] = image_postprocessing_type
model.config.foveated_config['selected_crop_size'] = selected_crop_size
model.config.foveated_config['token_total'] = token_total
model.config.foveated_config['gaze_alpha'] = gaze_beta
model.config.foveated_config['print_flag'] = False

#####################################################

question_id = '03627620'
model.config.foveated_config['fixations'] = [float(203.690265), float(373.982645)]
image_id = '2358679'
image = Image.open("./assets/"+str(image_id)+".jpg")
my_prompt = 'What is that bicycle leaning against?'

processor.image_processor.foveated_config = model.config.foveated_config

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": my_prompt},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(image, prompt, return_tensors="pt").to("cuda:0")

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=100)
output_text = processor.decode(output[0], skip_special_tokens=True)


print(output_text)
print()
