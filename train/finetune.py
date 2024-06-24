import os
import torch
import pandas as pd
from datasets import load_dataset,Dataset ,DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig,get_peft_model, prepare_model_for_int8_training,prepare_model_for_kbit_training
from trl import SFTTrainer 
from trl import DataCollatorForCompletionOnlyLM

from huggingface_hub import login
import yaml


print("Imported libs \n")


# Define a function to load parameters from a YAML file
def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load parameters from YAML file
config = load_config_from_yaml("train_config.yaml")
# Access parameters
model_path = config["model_path"]
new_model_path = config["new_model_path"]
dataset_path = config["dataset_path"]
instruction_token = config["instruction_token"]
Response_token = config["Response_token"]



tokenizer_pad_token = config["tokenizer_pad_token"]
tokenizer_pad_side = config["tokenizer_pad_side"]
load_in_4bit = config["load_in_4bit"]
load_in_8bit = config["load_in_8bit"]
lora_alpha_value = config["lora_alpha_value"]
lora_dropout_value = config["lora_dropout_value"]
target_modules_value = config["target_modules_value"]
r_value = config["r_value"]
bias_value = config["bias_value"]
num_train_epochs_value = config["num_train_epochs_value"]
micro_batch_size_value_ = config["micro_batch_size_value_"]
gradient_accumulation_steps_value = config["gradient_accumulation_steps_value"]
optim_value = config["optim_value"]
save_steps_value = config["save_steps_value"]
logging_steps_value = config["logging_steps_value"]
learning_rate_value = float(config["learning_rate_value"])
weight_decay_value = config["weight_decay_value"]
fp16_value = config["fp16_value"]
bf16_value = config["bf16_value"]
warmup_steps_value = config["warmup_steps_value"]
lr_scheduler_type_value = config["lr_scheduler_type_value"]
push_to_hub_value = config["push_to_hub_value"]
use_wandb = config["use_wandb"]
max_seq_length_value = config["max_seq_length_value"]
read_hub_token = config["read_hub_token"]
write_hub_token = config["write_hub_token"]
hf_saving_repo = config["hf_saving_repo"]
wandb_project = config["wandb_project"]
mask_input = config["mask_input"]



print("Loaded parameters \n")

os.environ["WANDB_PROJECT"]=wandb_project




login(token=read_hub_token)

new_model = new_model_path
base_model=model_path
dataset=load_dataset(dataset_path)


def update_text(text):
    text=text.replace("### Instruction:"," </s><s>\n### Instruction:\n")
    text=text.replace(" </s><s>\n### Instruction:\n","\### Instruction:",1)
    text=text.replace("### Response:","\n### Response:")     
    return text 

def replace_token(example):
  new_text = [update_text(item) for item in example["text"]]
  example["text"] = new_text
  return example
  
dataset["train"] = dataset["train"].map(replace_token, batched=True)

print("Dataset loaded \n")



compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

print("Quantization configured \n")


if load_in_4bit :
    model= AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map="auto"
    )
elif load_in_8bit :
    model= AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
    )
else:
    model= AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto"
    )


model.config.use_cache = False

print("Model loaded and configured \n")

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
print("Tokenizer loaded \n")
bos = tokenizer.bos_token_id
eos = tokenizer.eos_token_id
pad = tokenizer.pad_token_id
print("pre-trained model's BOS EOS and PAD token id:",bos,eos,pad," => It should be 1 2 None")

tokenizer.pad_token_id = tokenizer_pad_token  # unk. we want this to be different from the eos token
tokenizer.padding_side =tokenizer_pad_side


print("Tokenizer configered \n")


if mask_input :

    #response_template = "\n### Response:"
    response_template_with_context = f"\n{Response_token}"  # We added context here: "\n". This is enough for this tokenizer
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`
    instruction_template_with_context = f"\n{instruction_token}"  # We added context here: "\n". This is enough for this tokenizer
    instruction_template_ids = tokenizer.encode(instruction_template_with_context, add_special_tokens=False)[2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, instruction_template=instruction_template_ids,tokenizer=tokenizer,mlm=False, pad_to_multiple_of=8, return_tensors="pt")
else:
    collator=None


print("Collator configered \n")

peft_params = LoraConfig(
    lora_alpha=lora_alpha_value,
    lora_dropout=lora_dropout_value,
	target_modules=target_modules_value,
    r=r_value,
    bias=bias_value,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_params)

print("Peft configered \n")


if use_wandb:
    report_value="wandb"
else:
    report_value=None


training_params = TrainingArguments(
    num_train_epochs=num_train_epochs_value,
    per_device_train_batch_size=micro_batch_size_value_,
    gradient_accumulation_steps=gradient_accumulation_steps_value,
    optim=optim_value,
    save_steps=save_steps_value,
    logging_steps=logging_steps_value,
    learning_rate=learning_rate_value,
    weight_decay=weight_decay_value,
    fp16=fp16_value,
    bf16=bf16_value,
    warmup_steps=warmup_steps_value,
    group_by_length=False,
    lr_scheduler_type=lr_scheduler_type_value,
    output_dir=new_model_path,
    push_to_hub=push_to_hub_value,
    report_to = report_value
)


print("Training parameteres set \n")

print(f"Example of training data: \n {dataset['train'][1]['text']} \n")



trainer = SFTTrainer(
    model=model,
    args=training_params,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=max_seq_length_value,
    data_collator = collator,

)

print("Trainer set \n")

print("Training begins \n")

print("\n")

trainer.train()

print("\n")

print("Training ends \n")

# Push the trained model to Hugging Face Hub
if push_to_hub_value:
    login(token=read_hub_token)
    trainer.push_to_hub(f"{hf_saving_repo}")
    



trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

print("\nModel saved\n\n")

