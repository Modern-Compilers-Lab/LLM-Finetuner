import yaml
import os

# Define a function to load parameters from a YAML file
def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config_from_yaml("main.yaml")




if config['preprocess'] == True:
    os.system("python preprocess/preprocess.py")
    
if config['train'] == True:
    os.system("python train/finetune.py")

if config['generate'] == True:
    os.system("python generation/generation.py")     
    
if config['evaluate'] == True:
    os.system("python evaluate/evaluate.py")
    
           