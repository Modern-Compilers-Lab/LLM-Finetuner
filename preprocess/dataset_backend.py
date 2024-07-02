import yaml

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

def sequence_lengths(dataset_obj):

    # Initialize a list to store the sequence lengths
    sequence_lengths = []

    # list of indices that are too long
    too_long = []

    # Loop over the dataset and get the lengths of text sequences
    # for idx, example in enumerate(dataset_obj["train"]):
    for idx, example in enumerate(dataset_obj):
        sequence_lengths.append(len(example[output_column_name]))
        if sequence_lengths[idx] > sequence_length:
          too_long.append(idx)

    return too_long


def reduce_prompts_size(indexes,dataset_obj):
   return  dataset_obj.select(i for i in range(len(dataset_obj)) if i not in set(indexes))
   



def formatting_func(example):

   
   if context_column_name == False:
      prompt = f"{sys_prompt_no_context}\n\n{instruction_token}:\n{example[instruction_column_name]}\n\n{Response_token}:\n {example[response_column_name]} </s>"
   else: 
      if example[context_column_name] != '':
         prompt = f"{sys_prompt_context}\n\n{instruction_token}:\n{example[instruction_column_name]}\n\n{context_token}:\n{example[context_column_name]}\n\n{Response_token}:\n{example[response_column_name]} </s>"
      else :
         prompt = f"{sys_prompt_no_context}\n\n{instruction_token}:\n{example[instruction_column_name]}\n\n{Response_token}:\n {example[response_column_name]} </s>"


   return {output_column_name:prompt}