from huggingface_hub import login
from transformers import AutoTokenizer
from datasets import  DatasetDict, load_dataset, concatenate_datasets
from dataset_backend import formatting_func,reduce_prompts_size,sequence_lengths
import yaml

print("Imported libs \n")


# Open the YAML file
with open("preprocess/data_format.yaml", "r") as yaml_file:
    # Load the YAML data
    config = yaml.safe_load(yaml_file)

##Load configuration attributes

sequence_length = config["prompt_size"]

instruction_token = config["instruction_token"]
Response_token = config["Response_token"]
context_token = config["context_token"]

sys_prompt_no_context = config["system_prompt_no_context"]
sys_prompt_context = config["system_prompt_context"]

instruction_column_name = config["instruction_column_name"]
context_column_name = config["context_column_name"]
response_column_name = config["response_column_name"]
output_column_name = config["output_column_name"]

print("Loaded parameters \n")

if config["hf_token_read"]!= False:
    login(token=config["hf_token_read"])
else:
    print("no read token was specified you might face an error while loading the tokenizer")
    
dataset = load_dataset(config["dataset_hub_repo"])

print("Dataset loaded \n")

tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_hub_repo"])

print("Tokenizer loaded \n")
#keep only the needed columns
if context_column_name:
    dataset = dataset.select_columns([instruction_column_name, context_column_name, response_column_name])
else:
    dataset = dataset.select_columns([instruction_column_name, response_column_name])
        
print("Pre-processing begins \n")
##format dataset
dataset = dataset.map(formatting_func)

## resize prompts 
dataset['train'] = reduce_prompts_size(sequence_lengths(dataset['train']),dataset['train'])

if'test' in dataset.column_names:
   dataset['test'] = reduce_prompts_size(sequence_lengths(dataset['test']),dataset['test'])
   #flatten both parts train and test
   dataset = DatasetDict({"train": concatenate_datasets([dataset["train"], dataset["test"]])})

#split dataset back to train/test
dataset = dataset['train'].train_test_split(test_size = config['test_size'] , shuffle = True)


login(token=config["hf_token_write"])

dataset.push_to_hub(config["save_dataset_hub_repo"],token=True)

print("Dataset saved \n")