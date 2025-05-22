# This codebase is constructed on top of Transformers (https://github.com/huggingface/transformers)
# The edited or added codes are marked with 'GazeVLM'
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
token_total = 500   # total token budget
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
if image_preprocessing_type == True:
    selected_crop_size = [int((672//2) * crop_alpha), \
                    (672//2)] 
force_resolution = [672, 672]
#------------------------------------------------------------------------#
# fixation_file_name = './assets/AiR-D/fixation_csv/question_fixation_mean_correct.csv'
fixation_file_name = './assets/AiR-D/fixation_csv/SAMPLE-question_fixation_mean_correct.csv'
qa_file_name = './assets/AiR-D/GQA/questions/val_balanced_questions.json'
##########################################################################
##########################################################################
##########################################################################

processor = LlavaNextProcessor.from_pretrained(model_name)
model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.to("cuda:0")

qa_file = open(qa_file_name)
qa = json.load(qa_file)

fixation_file_t = open(fixation_file_name)
fixation_file = csv.reader(fixation_file_t)

model.config.foveated_config = { }
model.config.foveated_config['force_resolution'] = force_resolution
model.config.foveated_config['image_preprocessing_type'] = image_preprocessing_type
model.config.foveated_config['image_postprocessing_type'] = image_postprocessing_type
model.config.foveated_config['selected_crop_size'] = selected_crop_size
model.config.foveated_config['token_total'] = token_total
model.config.foveated_config['gaze_alpha'] = gaze_beta

correct = 0
wrong = 0
start_time = time.time()
answers = { }
memory_avg = []
latency_avg = []
ntoken_avg = []
throughput_avg = []

for question_fixation in fixation_file:
    start_time2 = time.time()
    torch.cuda.reset_peak_memory_stats("cuda:0")

    model.config.foveated_config['fixations'] = [float(question_fixation[2]), float(question_fixation[1])]
    processor.image_processor.foveated_config = model.config.foveated_config

    image_id = qa[question_fixation[0]]['imageId']
    image = Image.open("./assets/AiR-D/GQA/images/"+image_id+".jpg")

    my_prompt = qa[question_fixation[0]]['question']

    if 'individual' in fixation_file_name:
        key = question_fixation[0] + '-' + question_fixation[3]
    else:
        key = question_fixation[0]
    answers[key] = { }
    answers[key]['question'] = my_prompt
    answers[key]['imageID'] = qa[question_fixation[0]]['imageId']

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
    answers[key]['answer'] = qa[question_fixation[0]]['answer']
    answers[key]['fullAnswer'] = qa[question_fixation[0]]['fullAnswer']
    answers[key]['generatedAnswer'] = output_text

    end_time2 = time.time()
    peak_memory_used = torch.cuda.max_memory_allocated("cuda:0") / (1024 * 1024 * 1024)

    print(output_text)
    print()
    
    if model_name == "llava-hf/llama3-llava-next-8b-hf":
        output_text = output_text.split('assistant\n\n\n')[1]
    elif model_name == "llava-hf/llava-v1.6-vicuna-7b-hf":
        output_text = output_text.split('ASSISTANT:')[1]
    elif model_name == "llava-hf/llava-v1.6-mistral-7b-hf":
        output_text = output_text.split('[/INST]')[1]
    elif model_name == "llava-hf/llava-v1.6-vicuna-13b-hf":
        output_text = output_text.split('ASSISTANT:')[1]
        
    if qa[question_fixation[0]]['answer'].lower() in output_text.lower():
        correct = correct + 1
        answers[key]['correct'] = 1
    else:
        wrong = wrong + 1
        answers[key]['correct'] = 0

    memory_avg.append(peak_memory_used)
    latency_avg.append(end_time2 - start_time2)
    ntoken_avg.append(model.config.foveated_config['n_image_tokens'])
    throughput_avg.append(len(output) / (end_time2 - start_time2))

    answers[key]['latency'] = end_time2 - start_time2
    answers[key]['memory usage'] = peak_memory_used
    answers[key]['num image tokens'] = model.config.foveated_config['n_image_tokens']
    answers[key]['num total tokens'] = model.config.foveated_config['n_tokens_llm']
    answers[key]['throughput'] = len(output) / (end_time2 - start_time2)

end_time = time.time()

##########################################################################
##########################################################################
##########################################################################

# outfile_name = './results/answers_'\
#                 +model_name.split('/')[1]\
#                 +str(time.localtime().tm_mon)+'_'+str(time.localtime().tm_mday)+'_'\
#                 +str(time.localtime().tm_hour)+'_'+str(time.localtime().tm_min)+'.json'

# with open(outfile_name, 'w') as outfile:
#     json.dump(answers, outfile)
    
print('----------')
print('Accruacy: ', correct * 100.0 / (correct+wrong))
print('Latency: ', sum(latency_avg) / len(latency_avg))
print('Peak GPU memory used (GB): ', sum(memory_avg) / len(memory_avg))
print('# of Image Tokens: ', sum(ntoken_avg) / len(ntoken_avg))
print('Throughput (token/s): ', sum(throughput_avg) / len(throughput_avg))
print('----------')
