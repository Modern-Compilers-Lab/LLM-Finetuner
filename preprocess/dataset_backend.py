
from preprocess import sequence_length,context_token,instruction_token,Response_token
from preprocess import instruction_column_name,context_column_name,response_column_name,output_column_name
from preprocess import sys_prompt_context, sys_prompt_no_context

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