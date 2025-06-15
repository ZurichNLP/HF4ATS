import torch
from utils import sft_dependencies
from utils import base_dependencies
import torch.distributed as dist

import json
import wandb
import os

import sys
import pynvml
from accelerate import Accelerator

import atexit
import os
import signal

def cleanup():
    print("Cleaning up processes...")
    os.system('kill $(ps aux | grep "pt_elasti" | awk \'{print $2}\')')

# Register the cleanup function to be called on script exit
atexit.register(cleanup)

torch.cuda.empty_cache()

model_name = sys.argv[1]
paramix = sys.argv[2]

print(paramix)
bs, ga_steps, gpu_count, lr = base_dependencies.paramix_parser(paramix)

# auto searches for n gpus with most available memory and chooses them
base_dependencies.select_and_report_gpus(gpu_count)
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# for cuda issues
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

max_seq_length = 300

model, tokenizer = base_dependencies.load_model_and_tokenizer_wrapper(model_name, 'sft_train')

print(f"Model on GPU? {next(model.parameters()).is_cuda}")
print(f"Model dtype: {next(model.parameters()).dtype}")
sft_dependencies.print_trainable_parameters(model)

#wb_token = ""
#wandb.login(key=wb_token)

for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_properties(i).name)

train_data = base_dependencies.load_jsonl('data/sft_train.jsonl')
train_dataset = base_dependencies.create_huggingface_dataset(train_data)

print(f"Train dataset: {train_dataset}")

sft_dependencies.print_trainable_parameters(model)

is_cuda = next(model.parameters()).is_cuda
print(f"Model is on the GPU? {is_cuda}")

dtype = next(model.parameters()).dtype
print(f"Model dtype: {dtype}")

sft_dependencies.run_train_sfttrainer(model, tokenizer, train_dataset, max_seq_length, model_name, paramix)
