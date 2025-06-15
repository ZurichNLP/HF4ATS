import torch
from utils import base_dependencies
from utils import eval_dependencies

import os
import sys
import json

torch.cuda.empty_cache()

model_name = sys.argv[1]
eval_type = sys.argv[2]
paramix = sys.argv[3]
dev_or_test = sys.argv[4]

bs, ga_steps, gpu_count, lr = base_dependencies.paramix_parser(paramix)
print(eval_type, dev_or_test, 'eval for', model_name)
subdirectory = paramix + '/'
if dev_or_test == 'test':
    subdirectory = subdirectory + 'test/'

# auto searches for n gpus with most available memory and chooses them
base_dependencies.select_and_report_gpus(gpu_count)
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# for cuda issues
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# import holdout dataset
if dev_or_test == 'test':
    test_data = base_dependencies.load_jsonl('data/deplain/sft_holdout.jsonl') #[0:400]
else:
    test_data = base_dependencies.load_jsonl('data/deplain/sft_dev.jsonl') #[0:400]
test_dataset = base_dependencies.create_huggingface_dataset(test_data)
print(f"{dev_or_test} dataset: {test_dataset}")

# format original, generated, and reference
## original and reference sentences
orig_sents = [dict['original'] for dict in test_data]
ref_sents = [dict['completion'] for dict in test_data]

max_seq_length = 300

if "sft" not in model_name and "dpo" not in model_name:
    gen_sents = eval_dependencies.get_generations(model_name, test_dataset, '', 'sft_eval', max_seq_length)
    eval_dependencies.eval_slate(eval_type, model_name, orig_sents, ref_sents, gen_sents, test_dataset, '', max_seq_length)
elif "sft" in model_name and 'dpo' not in model_name and dev_or_test == 'dev':
    for model_version in os.listdir(f"outputs/models/{paramix}/" + model_name):
        print(model_version)
        gen_sents = eval_dependencies.get_generations(model_name + "_" + model_version, test_dataset, subdirectory, 'sft_eval', max_seq_length)
        eval_dependencies.eval_slate(eval_type, model_name + "_" + model_version, orig_sents, ref_sents, gen_sents, test_dataset, subdirectory, max_seq_length, ignore_equals = True)
elif dev_or_test == 'test':
    if model_name == "sft_disco_llama8b":
        checkpoint = '2800'
    if model_name == "sft_llama8b":
        checkpoint = '2400'
    if model_name == "sft_leolm_mistral7b":
        checkpoint = '1600'
    if model_name == "dpo_sft_disco_llama8b_checkpoint2800_dataall_ea":
        checkpoint = '-' + str(int(1080/8))
    if model_name == "dpo_sft_disco_llama8b_checkpoint2800_dataall_ta":
        checkpoint = '-' + str(int(2160/8))
    if model_name == "dpo_sft_llama8b_checkpoint2400_dataall_ea":
        checkpoint = '-' + str(int(1320/8))
    if model_name == "dpo_sft_llama8b_checkpoint2400_dataall_ta":
        checkpoint = '-' + str(int(1440/8))
    if model_name == "dpo_sft_leolm_mistral7b_checkpoint1600_dataall_ea":
        checkpoint = '-' + str(int(2280/8))
    if model_name == "dpo_sft_leolm_mistral7b_checkpoint1600_dataall_ta":
        checkpoint = '-' + str(int(1560/8))

    gen_sents = eval_dependencies.get_generations(model_name + "_checkpoint" + checkpoint, test_dataset, subdirectory, 'dpo_eval', max_seq_length, subdirectory_fixer = True)
    eval_dependencies.eval_slate(eval_type, model_name + "_checkpoint" + checkpoint, orig_sents, ref_sents, gen_sents, test_dataset, subdirectory, max_seq_length, dev_or_test, ignore_equals = True)
