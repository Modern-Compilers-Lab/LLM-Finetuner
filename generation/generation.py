import json
import pandas as pd
from peft import AutoPeftModelForCausalLM
from transformers import GenerationConfig
from transformers import AutoTokenizer
import torch
import tqdm
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig,BitsAndBytesConfig
from datasets import load_dataset,Dataset ,DatasetDict,load_from_disk
import time
import re
from peft import PeftModel
import yaml

print("Imported libs \n")

# Define a function to load parameters from a YAML file
def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load parameters from YAML file
config = load_config_from_yaml("generation/generate_config.yaml")

# Access parameters
dataset_path = config["dataset_path"]
model_id = config["model_path"]
updated_dataset_path = config["updated_dataset_path"]
tokenizer_pad_token = config["tokenizer_pad_token"]
tokenizer_pad_side = config["tokenizer_pad_side"]
load_in_4bit = config["load_in_4bit"]
load_in_8bit = config["load_in_8bit"]
do_sample_value = config["do_sample_value"]
num_beams_value = config["num_beams_value"]
num_return_sequences_value = config["num_return_sequences_value"]
max_new_tokens_value = config["max_new_tokens_value"]
temperature_value = config["temperature_value"]
top_p_value = config["top_p_value"]
inferance_batch_size = config["inferance_batch_size"]
hub_token = config["hub_token"]

print("Loaded parameters \n")

login(token=hub_token)




tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = 0 
tokenizer.padding_side ="left"




print("tokenizer configered")


dataset_p=load_dataset(dataset_path)

val_data =dataset_p['test']



val_data_prompts = list(val_data['text'])
for i in range(len(val_data_prompts)):
    parts= val_data_prompts[i].split("\n### Response:\n")
    val_data_prompts[i]=parts[0]+"\n### Response:\n"

print("Dataset loaded \n")

compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

print("Quantization configured \n")

# Load the model and the tokenizer. Set generation config

if load_in_4bit :
    model= AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map="auto"
    )
elif load_in_8bit :
    model= AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
    )
else:
    model= AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto"
    )

print("Model loaded and configured \n")


generation_config = GenerationConfig(
    do_sample=do_sample_value,
	num_beams=num_return_sequences_value, 
	num_return_sequences=num_return_sequences_value,
	max_new_tokens=max_new_tokens_value,
	temperature=temperature_value,
	top_p=top_p_value,
	pad_token_id=tokenizer_pad_token
)


print("Generation config set \n")

def solve_question(question_prompt):
    inputs = tokenizer(question_prompt, return_tensors="pt", padding=True, truncation=True,max_length= 2048).to("cuda")
    outputs = model.generate(**inputs, generation_config=generation_config)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return answer




print("Inference starts\n")



all_answers = []
for i in range(0, len(val_data_prompts), inferance_batch_size):
    print("iteration: ", i)
    question_prompts = val_data_prompts[i:i+inferance_batch_size]
    
    # Record start time
    start_time = time.time()
    
    ans = solve_question(question_prompts)
    all_answers.extend(ans)
    
    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Print execution time for the iteration
    print("iteration time: {:.4f} seconds".format(execution_time))
    
    # Clear CUDA memory cache
    torch.cuda.empty_cache()


val_data=val_data.add_column("finetuned_answers",all_answers)



val_data.save_to_disk(updated_dataset_path)


print("Data saved \n")