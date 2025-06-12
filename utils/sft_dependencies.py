import torch
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from trl.trainer import ConstantLengthDataset
from datasets import load_dataset, Dataset
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from peft import get_peft_model, LoraConfig, PeftModel, PeftConfig, AutoPeftModelForCausalLM
from huggingface_hub import login
from math import ceil
import torch.distributed as dist
from accelerate import Accelerator
from utils import base_dependencies

import json
import wandb
import os
import shutil

################################################################################ TRAIN SFTTRAINER

def print_tokens_with_ids(tokenizer, txt):
    tokens = tokenizer.tokenize(txt, add_special_tokens=False)
    token_ids = tokenizer.encode(txt, add_special_tokens=False)
    print(list(zip(tokens, token_ids)))

def test_response_format(example, tokenizer, response_format):
    print("\nTest response format")
    example['prompt'] = [example['prompt']]
    example['completion'] = [example['completion']]
    print_tokens_with_ids(tokenizer, base_dependencies.formatting_prompts_func(example)[0])
    print('\n')
    print_tokens_with_ids(tokenizer, response_format)

def run_train_sfttrainer(model, tokenizer, train_dataset, max_seq_length, model_name, paramix):

    bs, ga_steps, gpu_count, lr = base_dependencies.paramix_parser(paramix)
    
    assert gpu_count == torch.cuda.device_count(), f"GPU count mismatch: {gpu_count} != {torch.cuda.device_count()}"

    tmp_dir = "outputs/tmp/"
    accelerator = Accelerator()

    if accelerator.is_local_main_process:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    peft_config = LoraConfig(r=16,
                            lora_alpha=32,
                            lora_dropout=0.05,
                            bias="none",
                            task_type="CAUSAL_LM",
                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"])

    per_device_train_batch_size = bs
    gradient_accumulation_steps = ga_steps
    
    ebs = per_device_train_batch_size * gradient_accumulation_steps * gpu_count
    #steps_per_epoch = len(train_dataset) // (ebs)
    
    checkpoint_save_interval = 400
    steps_per_checkpoint = checkpoint_save_interval // ebs # common multiple of 16, 32, and 64 close to 800
    

    training_args = SFTConfig(
        output_dir=tmp_dir,
        num_train_epochs=1,
        gradient_accumulation_steps = gradient_accumulation_steps,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=4,
        warmup_ratio=0.1,
        learning_rate=float(lr)*gpu_count,
        group_by_length=True,
        fp16=True,
        optim="paged_adamw_32bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        report_to="wandb",
        max_grad_norm=1,
        logging_strategy="no",
        eval_strategy="no",
        max_seq_length=max_seq_length,
        run_name="model_name_sft",
        save_strategy="steps",
        save_steps = steps_per_checkpoint
    )

    wandb.init(project="mnlp", name=model_name)

    if 'llama' in model_name.lower(): #and 'disco' not in model_name.lower()
        print('for completion only')
        trainer = SFTTrainer(
            model = model, 
            tokenizer = tokenizer,
            train_dataset = train_dataset,
            eval_dataset = None,
            args = training_args,
            formatting_func = base_dependencies.formatting_prompts_func,
            data_collator = base_dependencies.load_collator(tokenizer),
            peft_config=peft_config
        )
    else:
        print('next token pred')
        trainer = SFTTrainer(
            model = model, 
            tokenizer = tokenizer,
            train_dataset = train_dataset,
            eval_dataset = None,
            args = training_args, # formatting_func = formatting_prompts_func, # data_collator = load_collator(tokenizer),
            peft_config=peft_config
        )

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print("\nInitialized SFT Trainer...")
    try:
        print(f'pad token: {tokenizer.pad_token}')
    except:
        print('no pad token')
    
    try:
        print(f'pad side: {tokenizer.padding_side}')
    except:
        print('no padding side')

    if 'llama' in model_name.lower() and 'disco' not in model_name.lower():
        test_response_format(train_dataset[0], tokenizer, response_format = " ### assistant:")

    print("\nStart training...")

    torch.cuda.empty_cache()

    model.train()
    trainer.train()

    print("\nTraining completed.")

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    accelerator.wait_for_everyone()

    #dev_data = base_dependencies.load_jsonl('data/sft_dev.jsonl')[0:400]
    #dev_dataset = base_dependencies.create_huggingface_dataset(dev_data)
    #tokenizer.padding_side = 'left'
    #base_dependencies.generate_output(model, model_name, tokenizer, dev_dataset, "bs4ga2dv1lr1e-4/", max_seq_length, save = True)
    #tokenizer.padding_side = 'right'
    #base_dependencies.generate_output(model, model_name, tokenizer, dev_dataset, "bs4ga2dv1rr1e-4/", max_seq_length, save = True)
    
    if accelerator.is_local_main_process:

        base_dependencies.print_trainable_parameters(model)
        print(tmp_dir + model_name)
        trainer.save_model(tmp_dir + model_name)
        tokenizer.save_pretrained(tmp_dir + model_name)
    
        #trainer.save_model('outputs/adaptors/' + f'bs{per_device_train_batch_size}ga{gradient_accumulation_steps}dv{gpu_count}/sft_{model_name}/base_{model_name}/')
        #tokenizer.save_pretrained('outputs/adaptors/' + f'bs{per_device_train_batch_size}ga{gradient_accumulation_steps}dv{gpu_count}/sft_{model_name}/base_{model_name}/')

        del model
        del trainer
        del tokenizer
        #del state_dict

    for model_version in os.listdir(tmp_dir):

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        accelerator.wait_for_everyone()

        if accelerator.is_local_main_process:
            
            if "checkpoint" in model_version:
                print(model_version)

            #     new_checkpoint_name = f"checkpoint{int(model_version.split('-')[1]) * (gpu_count * per_device_train_batch_size * gradient_accumulation_steps)}"
            #     new_folder_name = 'outputs/adaptors/' + f'bs{per_device_train_batch_size}ga{gradient_accumulation_steps}dv{gpu_count}/sft_{model_name}/'
            #     shutil.move(tmp_dir + model_version, new_folder_name)
            #     os.rename(new_folder_name + model_version, new_folder_name + new_checkpoint_name)

                checkpoint_num = int(model_version.split('-')[1]) * (gpu_count * per_device_train_batch_size * gradient_accumulation_steps)
                model_checkpoint_name = f"checkpoint{checkpoint_num}"
                print(checkpoint_num)
                if checkpoint_num % checkpoint_save_interval == 0: #and checkpoint_num <= checkpoint_save_interval*2
                    param_mix_direc = f"outputs/models/{paramix}/"
                    if not os.path.exists(param_mix_direc):
                        os.makedirs(param_mix_direc)
                    if not os.path.exists(param_mix_direc + 'sft_' + model_name + '/' + model_checkpoint_name):

                        model = AutoPeftModelForCausalLM.from_pretrained(
                            tmp_dir + model_version,
                            torch_dtype=torch.float16,
                            low_cpu_mem_usage=True,
                        )

                        tokenizer = AutoTokenizer.from_pretrained(tmp_dir + model_version)
                        merged_model = model.merge_and_unload()
                        model_checkpoint_name = f"checkpoint{checkpoint_num}"
                        print(model_version, model_name, model_checkpoint_name, gpu_count)
                        merged_model.save_pretrained(f"outputs/models/{paramix}/sft_" + model_name + "/" + model_checkpoint_name)
                        tokenizer.save_pretrained(f"outputs/models/{paramix}/sft_" + model_name + "/" + model_checkpoint_name)
            else:
                print(model_version)

    accelerator.wait_for_everyone()
    