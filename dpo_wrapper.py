import torch
from utils import sft_dependencies
from utils import base_dependencies
from utils import dpo_dependencies
import torch.distributed as dist

import random
import json
import wandb
import os
import pandas as pd
import sys
import pynvml
from accelerate import Accelerator

import atexit
import os
import signal

print('starting dpo')
#torch.cuda.empty_cache()

model_name = sys.argv[1]
variant = sys.argv[2]
user_set = sys.argv[3]
train_or_test = sys.argv[4]

paramix = 'bs16ga1dv1lr1e-4'
print(model_name, variant, paramix)
bs, ga_steps, gpu_count, lr = base_dependencies.paramix_parser(paramix)
base_dependencies.select_and_report_gpus(gpu_count)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if model_name not in ['sft_disco_llama8b_checkpoint2800', 'sft_llama8b_checkpoint2400', 'sft_leolm_mistral7b_checkpoint1600']:
    raise ValueError("Invalid model for inference. Must be a winning checkpoint ('sft_disco_llama8b_checkpoint2800', 'sft_llama8b_checkpoint2400', 'sft_leolm_mistral7b_checkpoint1600')")
if variant not in ['all', 'equality', 'model', 'intraAA', 'interAA', 'groupX']:
    raise ValueError("Invalid data variant.")
model_name2 = 'bs16ga1dv1lr1e-4/' + model_name

if train_or_test == 'train':
    sub_fp = 'dev'
if train_or_test == 'test':
    sub_fp = 'test'
variant_completed = os.path.exists(f"outputs/dpo_{sub_fp}_eval/trainer_state_{model_name}_data{variant}_{user_set}.json")
skippable_variant = (variant == 'groupX') & (user_set == 'ta')

if not variant_completed and not skippable_variant:
        
    print('loading model')
    model, tokenizer = base_dependencies.load_model_and_tokenizer_wrapper(model_name2, 'dpo_train')
    base_dependencies.select_and_report_gpus(1)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    max_seq_length = 300

    print(f"Model on GPU? {next(model.parameters()).is_cuda}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    base_dependencies.print_trainable_parameters(model)

    #wb_token = ""
    #wandb.login(key=wb_token)

    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)

    if variant == 'all':
        train_data = base_dependencies.load_jsonl(f'data/preferences/preferences_all.jsonl')
    if variant == 'equality':
        train_data = base_dependencies.load_jsonl(f'data/preferences/preferences_ie.jsonl')
    if variant == 'model':
        if 'disco' in model_name:
            train_data = base_dependencies.load_jsonl(f'data/preferences/preferences_sft_disco_llama8b_checkpoint2800.jsonl')
        if 'llama' in model_name and 'disco' not in model_name:
            train_data = base_dependencies.load_jsonl(f'data/preferences/preferences_sft_llama8b_checkpoint2400.jsonl')
        if 'mistral' in model_name:
            train_data = base_dependencies.load_jsonl(f'data/preferences/preferences_sft_leolm_mistral7b_checkpoint1600.jsonl')
    if variant == 'intraAA':
        train_data = base_dependencies.load_jsonl(f'data/preferences/preferences_itra.jsonl')
    if variant == 'interAA':
        train_data = base_dependencies.load_jsonl(f'data/preferences/preferences_iter.jsonl')
    if variant == 'groupX':
        train_data = base_dependencies.load_jsonl(f'data/preferences/preferences_all.jsonl')
        test_data = base_dependencies.load_jsonl(f'data/preferences/preferences_itra.jsonl')

    if variant != 'groupX':
        train_data = [single_pref for single_pref in train_data if single_pref['userGroup'] == user_set]
        train_split_idx = int(len(train_data) * 0.8)
        dev_split_idx = int(len(train_data) * 0.9)
        train_set, test_set, dev_set = train_data[:train_split_idx], train_data[train_split_idx:dev_split_idx], train_data[dev_split_idx:]
    else:
        train_set = [single_pref for single_pref in train_data if single_pref['userGroup'] == 'ea']
        dev_and_test_set = [single_pref for single_pref in test_data if single_pref['userGroup'] == 'ta']
        test_set = dev_and_test_set[0:int(len(dev_and_test_set) * 0.5)]
        dev_set = dev_and_test_set[int(len(dev_and_test_set) * 0.5):]

    train_originals = {item['original'] for item in train_set}
    dev_originals = {item['original'] for item in dev_set}
    test_originals = {item['original'] for item in test_set}

    if variant != 'groupX':
        assert train_originals.isdisjoint(dev_originals), "Overlap found between train and dev sets!"
        assert train_originals.isdisjoint(test_originals), "Overlap found between train and test sets!"
    assert dev_originals.isdisjoint(test_originals), "Overlap found between dev and test sets!"

    train_dataset = base_dependencies.create_huggingface_dataset(train_set, style='preference')
    dev_dataset = base_dependencies.create_huggingface_dataset(dev_set, style='preference')
    test_dataset = base_dependencies.create_huggingface_dataset(test_set, style='preference')

    print(f"Train dataset: {train_dataset}")
    print(f"Dev dataset: {dev_dataset}")
    print(f"Test dataset: {test_dataset}")

    if variant == 'all':
        test_save = f'data/preferences/test_{user_set}_preferences_all.jsonl'
    if variant == 'equality':
        test_save = f'data/preferences/test_{user_set}_preferences_ie.jsonl'
    if variant == 'model':
        if 'disco' in model_name:
            test_save = f'data/preferences/test_{user_set}_preferences_sft_disco_llama8b_checkpoint2800.jsonl'
        if 'llama' in model_name and 'disco' not in model_name:
            test_save = f'data/preferences/test_{user_set}_preferences_sft_llama8b_checkpoint2400.jsonl'
        if 'mistral' in model_name:
            test_save = f'data/preferences/test_{user_set}_preferences_sft_leolm_mistral7b_checkpoint1600.jsonl'
    if variant == 'intraAA':
        test_save = f'data/preferences/test_{user_set}_preferences_itra.jsonl'
    if variant == 'interAA':
        test_save = f'data/preferences/test_{user_set}_preferences_iter.jsonl'
    if variant == 'groupX':
        test_save = f'data/preferences/test_{user_set}_preferences_X.jsonl'

    with open(test_save, "w") as f:
        for item in test_set:
            f.write(json.dumps(item) + "\n")

    base_dependencies.print_trainable_parameters(model)

    is_cuda = next(model.parameters()).is_cuda
    print(f"Model is on the GPU? {is_cuda}")

    dtype = next(model.parameters()).dtype
    print(f"Model dtype: {dtype}")

    save_strategy = 'steps'
    dpo_dependencies.run_train_dpotrainer(model, tokenizer, train_dataset, dev_dataset, test_dataset, max_seq_length, model_name, variant, user_set, paramix, train_or_test, save_strategy)