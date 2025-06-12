import torch
from trl import SFTConfig, SFTTrainer
from trl.trainer import ConstantLengthDataset
from datasets import load_dataset, Dataset
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from peft import get_peft_model, LoraConfig, PeftModel, PeftConfig, AutoPeftModelForCausalLM
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from huggingface_hub import login
from accelerate import PartialState
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, DistilBertForSequenceClassification, DistilBertModel
import pandas as pd
from transformers import set_seed

import json
import wandb
import os
import shutil
import tqdm
import torch
from transformers import pipeline
import random
import pynvml
from accelerate import Accelerator
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import seaborn as sns

################################################################################ PARALLELISM

def get_device_string():
    device_string = PartialState().process_index
    print('device string:', device_string)
    return device_string

def select_and_report_gpus(gpu_count=1):
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    gpu_memory = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory.append((i, mem_info.free))
    sorted_gpus = sorted(gpu_memory, key=lambda x: x[1], reverse=True)
    selected_gpus = sorted_gpus[:gpu_count]
    selected_gpu_ids = [str(gpu[0]) for gpu in selected_gpus]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_gpu_ids)
    print(f"Selected GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")

    for gpu_id, memory in selected_gpus:
        print(f"GPU {gpu_id}: {memory / (1024**2):.2f} MB available")
    pynvml.nvmlShutdown()

################################################################################ PARAMIX & PARAMETER COUNT

def paramix_parser(paramix):
    print(paramix)
    bs = int(paramix[len('bs'):paramix.find('ga')])
    print(bs)
    ga_steps = int(paramix[paramix.find('ga') + 2:paramix.find('dv')])
    print(ga_steps)
    gpu_count = int(paramix[paramix.find('dv') + 2:paramix.find('lr')])
    print(gpu_count)
    lr = paramix[paramix.find('lr')+2:]
    print(lr)
    return bs, ga_steps, gpu_count, lr

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * (trainable_params / all_param)}"
    )

################################################################################ FUNC FOR SFT COMPLETION ONLY

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"### user: {example['prompt'][i]}\n ### assistant: {example['completion'][i]}"
        output_texts.append(text)
    return output_texts

def load_collator(tokenizer):
    response_template = " ### assistant:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, mlm=False)
    return collator

################################################################################ ROUGE

def pull_rouge(originals, simplifications):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    results = []
    for i in range(len(originals)):
        scores = scorer.score(originals[i], simplifications[i])  # Reference, Hypothesis
        results.append(scores)
    return results

def chart_rouge(results):
    rouge1_precision = [dct['rouge1'][0] for dct in results]
    rouge1_recall = [dct['rouge1'][1] for dct in results]
    rouge1_f1s = [dct['rouge1'][2] for dct in results]

    rouge2_precision = [dct['rouge2'][0] for dct in results]
    rouge2_recall = [dct['rouge2'][1] for dct in results]
    rouge2_f1s = [dct['rouge2'][2] for dct in results]

    rougeL_precision = [dct['rougeL'][0] for dct in results]
    rougeL_recall = [dct['rougeL'][1] for dct in results]
    rougeL_f1s = [dct['rougeL'][2] for dct in results]

    data_lists = [rouge1_precision, rouge1_recall, rouge1_f1s, rouge2_precision, rouge2_recall, rouge2_f1s, rougeL_precision, rougeL_recall, rougeL_f1s]
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    #fig.suptitle("ROUGE Score Distributions: Original vs Simplified Sentence", fontsize=14, fontweight='bold')
    axes = axes.flatten()
    for i, data in enumerate(data_lists):
        axes[i].set_ylim([0, 1550])
        axes[i].set_yticks([0, 500, 1000, 1500])
        axes[i].grid(True, axis='y', linestyle='-', linewidth=0.5, alpha=0.7)
        sns.histplot(data, kde=True, color='blue', bins=20, edgecolor='black', ax=axes[i], zorder=3)
        axes[i].set_xlim(0, 1)
        axes[i].set_title(f'Histogram {i+1}')
        if i in [2, 5, 8]:  
            axes[i].axvline(x=0.8, linestyle='--', color='red', linewidth=1, zorder=4)

    axes[0].set_title('Rouge-1 Precision')
    axes[1].set_title('Rouge-1 Recall')
    axes[2].set_title('Rouge-1 F1')
    axes[3].set_title('Rouge-2 Precision')
    axes[4].set_title('Rouge-2 Recall')
    axes[5].set_title('Rouge-2 F1')
    axes[6].set_title('Rouge-L Precision')
    axes[7].set_title('Rouge-L Recall')
    axes[8].set_title('Rouge-L F1')
    for j in range(len(data_lists), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig('figures/deplain_presft_rouge.png', dpi=800)
    plt.show()


################################################################################ EMBEDDING MODEL

def launch_embedding_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stemodel = SentenceTransformer('T-Systems-onsite/cross-en-de-roberta-sentence-transformer', device=device)
    return stemodel

def embeddings(list, stemodel):
    embeddings = [stemodel.encode(text) for text in list]
    return embeddings

def cossim(orig_embed, simp_embed):
    cossim = cosine_similarity([orig_embed], [simp_embed])
    return cossim[0][0]

def cossim_sentence_lists(stemodel, lst1, lst2):
    lst1_embeddings = embeddings(lst1, stemodel)
    lst2_embeddings = embeddings(lst2, stemodel)
    return [cossim(s1, s2) for s1,s2 in zip(lst1_embeddings, lst2_embeddings)]

################################################################################ ENTAILMENT MODEL

def launch_entailment_model():
    model_name = "svalabs/gbert-large-zeroshot-nli" #"MoritzLaurer/ernie-m-large-mnli-xnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(torch.device("cuda"))
    return model, tokenizer

def entailment(entail_model, entail_tokenizer, s1, s2, device):
    input = entail_tokenizer(s1, s2, truncation=True, return_tensors="pt")
    output = entail_model(input["input_ids"].to(device))
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label_names = ["neutral", "entailment", "contradiction"]
    return {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}

def entail_sentence_lists(entail_model, entail_tokenizer, lst1, lst2):
    return [entailment(entail_model, entail_tokenizer, s1, s2, torch.device("cuda")) for s1,s2 in zip(lst1, lst2)]

#base_dependencies.entailment(entail_model, entail_tokenizer, 'Van der Bellen sagte auch, dass die Österreicher sehr gut Hände waschen sollen.', 'Er sagte auch, dass die Menschen in Österreich die Hände waschen sollen.', torch.device("cuda"))
#base_dependencies.entailment(entail_model, entail_tokenizer, 'Van der Bellen sagte auch, dass die Österreicher sehr gut Hände waschen sollen.', 'Er sagte auch, dass die Menschen in Österreich mit der Bahn fahren sollten.', torch.device("cuda"))
#base_dependencies.entailment(entail_model, entail_tokenizer, 'Van der Bellen sagte auch, dass die Österreicher sehr gut Hände waschen sollen.', 'Er sagte auch, dass die Menschen in Österreich sich nicht die Hände waschen sollten.', torch.device("cuda"))
#base_dependencies.entailment(entail_model, entail_tokenizer, 'Van der Bellen sagte auch, dass die Österreicher sehr gut Hände waschen sollen.', 'Er hat auch gesagt, dass die Menschen in Österreich duschen sollten.', torch.device("cuda"))
#base_dependencies.entailment(entail_model, entail_tokenizer, 'Van der Bellen sagte auch, dass die Österreicher sehr gut Hände waschen sollen.', 'Er sagte auch, dass die Menschen in Österreich sich wahrscheinlich die Hände waschen sollten.', torch.device("cuda"))

################################################################################ TRANSLATION MODEL

def launch_translation_model():
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2ForTextToText.from_pretrained("facebook/seamless-m4t-v2-large", device_map={'':get_device_string()}, low_cpu_mem_usage=True)
    return model, processor

def translate_german_sentences(model, processor, sent_list, batch_size):
    english_sent_list = []
    for i in range(0, len(sent_list), batch_size):
        batch = sent_list[i:i+batch_size]
        text_inputs = processor(text = batch, src_lang="deu", return_tensors="pt").to(model.device)
        output_tokens = model.generate(**text_inputs, tgt_lang="eng")
        english_sent_list = english_sent_list + [processor.decode(output_token_tensor.tolist(), skip_special_tokens=True) for output_token_tensor in output_tokens]
    return english_sent_list

################################################################################ COMPLEXITY MODEL

def launch_complexity_model():
    model = DistilBertForSequenceClassification.from_pretrained('MiriUll/distilbert-german-text-complexity', device_map={'':get_device_string()}, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained('MiriUll/distilbert-german-text-complexity')
    return model, tokenizer

def complexity_sentence_list(complexity_model, complexity_tokenizer, sent_list, batch_size=32):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_preds = []

    for i in range(0, len(sent_list), batch_size):
        batch_sentences = sent_list[i:i+batch_size]
        inputs = complexity_tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Move to GPU if available
    
        with torch.no_grad():
            outputs = complexity_model(**inputs)
            predictions = outputs.logits.squeeze().cpu().numpy()

        all_preds.extend(predictions if predictions.ndim > 0 else [predictions])
        
    return all_preds

################################################################################ DATASET

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def create_huggingface_dataset(data, print_structure = False, style = 'task'):
    if style == 'task':
        prompts = [entry['prompt'] for entry in data]
        try:
            chosens = [entry['completion'] for entry in data]
        except:
            chosens = prompts
        dataset_dict = {
            'prompt': prompts,
            'completion': chosens
        }
        dataset = Dataset.from_dict(dataset_dict)
        if print_structure:
            print(f"{dataset}")
        return dataset
    elif style == 'preference':
        prompts = [entry['prompt'] for entry in data]
        chosens = [entry['chosen'] for entry in data]
        rejecteds = [entry['rejected'] for entry in data]
        dataset_dict = {
            'prompt': prompts,
            'chosen': chosens,
            'rejected': rejecteds
        }
        dataset = Dataset.from_dict(dataset_dict)
        if print_structure:
            print(f"{dataset}")
        return dataset

def load_model_and_tokenizer_wrapper(model_name, purpose):

    if model_name == "qwen3b":
        model_path = "Qwen/Qwen2.5-3B-Instruct"
    if model_name == "qwen7b":
        model_path = "Qwen/Qwen2.5-7B-Instruct"

    if model_name == "llama3b":
        model_path = "meta-llama/Llama-3.2-3B-Instruct"
    if model_name == "llama8b":
        model_path = "meta-llama/Llama-3.1-8B-Instruct"
    if model_name == "disco_llama8b":
        model_path = "DiscoResearch/Llama3-DiscoLeo-Instruct-8B-v0.1"

    if model_name == "leolm_mistral7b":
        model_path = "LeoLM/leo-mistral-hessianai-7b-chat"
    if model_name == "mistral7b":
        model_path = "mistralai/Mistral-7B-Instruct-v0.3"

    if model_name == "distillqwen7b":
        model_path = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'

    if "sft" in model_name:
        if "checkpoint" in model_name:
            model_path = "outputs/models/" + model_name[::-1].replace('_', '/', 1)[::-1] + '/'
        else:
            model_path = "outputs/models/" + model_name + "/"

    print(model_name)
    print(model_path)

    print("sft train: ", model_path)
    
    model, tokenizer = load_model_and_tokenizer(model_path, purpose)

    return model, tokenizer

def load_model_and_tokenizer(model_path, purpose):
    
    if 'dpo' not in model_path.lower() and 'sft' not in model_path.lower():
        login("<YOUR HUGGINGFACE TOKEN HERE>")  # RESEARCHER MUST REPLACE WITH HUGGINGFACE TOKEN
        
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map={'':get_device_string()}, low_cpu_mem_usage=True)
    
    print(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # remnant of attempt to use LeoLM/leo-hessianai-7b
    #if "leo" in model_path.lower():
        # taken from https://huggingface.co/jphme/em_german_leo_mistral/commit/4f03ad4ce09f81cdb00e20649cb0c91625fe26ce
        #chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{% if message['role'] == 'system' %}{{message['content'] + ' '}}{% elif message['role'] == 'user' %}{{ 'USER: ' + message['content'] + ' '}}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + ' '}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT: '}}{% endif %}"
        #tokenizer.chat_template = chat_template

    if purpose == 'sft_eval' or purpose == 'dpo_inference' or purpose == 'dpo_eval':
        print(purpose)

        # llama base
        if 'llama' in model_path.lower() and 'sft' not in model_path.lower() and 'disco' not in model_path.lower():
            print('base llama')
            print(model_path, "has no padding token. Setting to eos token.")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
        # sft llama
        elif 'llama' in model_path.lower() and ('sft' in model_path.lower() or 'dpo' in model_path.lower()) and 'disco' not in model_path.lower():
            print('sft llama')
            print(model_path, "has no padding token. Setting to special token.")
            tokenizer.add_special_tokens({"pad_token": "<|finetune_right_pad_id|>"})
            #tokenizer.pad_token = "<|finetune_right_pad_id|>"
            model.config.pad_token_id = tokenizer.pad_token_id
            tokenizer.padding_side = 'left'

        # disco llama base
        elif 'disco' in model_path.lower() and 'sft' not in model_path.lower():
            print('base disco')
            tokenizer.padding_side = 'left'
            pass
        # sft disco
        elif 'disco' in model_path.lower() and ('sft' in model_path.lower() or 'dpo' in model_path.lower()):
            print('sft disco')
            #print(model_path, "has no padding token. Setting to special token.")
            #tokenizer.add_special_tokens({"pad_token": "<|finetune_right_pad_id|>"})
            #tokenizer.pad_token = "<|finetune_right_pad_id|>"
            #model.config.pad_token_id = tokenizer.pad_token_id
            tokenizer.padding_side = 'left'

        # mistral base
        if 'mistral' in model_path.lower() and 'sft' not in model_path.lower() and 'leo' not in model_path.lower():
            print('base mistral')
            print(model_path, "has no padding token. Setting to special token.")
            tokenizer.add_special_tokens({'pad_token': '<unk>'})
            tokenizer.padding_side = 'left'
            pass
        # sft mistral
        elif 'mistral' in model_path.lower() and ('sft' in model_path.lower() or 'dpo' in model_path.lower()) and 'leo' not in model_path.lower():
            print('sft mistral')
            print(model_path, "has no padding token. Setting to special token.")
            tokenizer.add_special_tokens({'pad_token': '<unk>'})
            tokenizer.pad_token = '<unk>'
            model.config.pad_token_id = tokenizer.pad_token_id
            #tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'

        # leolm mistral base
        if 'mistral' in model_path.lower() and 'sft' not in model_path.lower() and 'leo' in model_path.lower():
            print('leolm base')
            tokenizer.padding_side = 'left'
            pass
        # sft leolm mistral
        elif 'mistral' in model_path.lower() and ('sft' in model_path.lower() or 'dpo' in model_path.lower()) and 'leo' in model_path.lower():
            print('sft leolm')
            print(model_path, "has no padding token. Setting to special token.")
            tokenizer.add_special_tokens({'pad_token': '<unk>'})
            model.config.pad_token_id = tokenizer.pad_token_id
            tokenizer.padding_side = 'left'

        # distill deepseek
        if 'distill' in model_path.lower():
            print('distill deepseek')
            tokenizer.padding_side = 'left'

    if purpose == 'sft_train' or purpose == 'dpo_train':
        print(purpose)
        # llama base
        if 'llama' in model_path.lower() and 'disco' not in model_path.lower():
            print('sft llama')
            print(model_path, "Setting pad to special token.")
            tokenizer.add_special_tokens({"pad_token": "<|finetune_right_pad_id|>"})
            model.config.pad_token_id = tokenizer.pad_token_id
            tokenizer.padding_side = 'right'

        # disco llama base
        elif 'disco' in model_path.lower() and tokenizer.pad_token is None:
            print('sft disco')
            print(model_path, "Setting pad to special token.")
            tokenizer.add_special_tokens({"pad_token": "<|finetune_right_pad_id|>"})
            model.config.pad_token_id = tokenizer.pad_token_id
            tokenizer.padding_side = 'right'

        # mistral base
        if 'mistral' in model_path.lower() and 'sft' not in model_path.lower() and 'leo' not in model_path.lower():
            print('base mistral')
            print(model_path, "Setting pad to special token.")
            tokenizer.add_special_tokens({'pad_token': '<unk>'})
            #tokenizer.pad_token = tokenizer.eos_token
            #tokenizer.pad_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.pad_token_id
            tokenizer.padding_side = 'right'

        # leolm mistral base
        if 'mistral' in model_path.lower() and 'sft' not in model_path.lower() and 'leo' in model_path.lower():
            print('leolm base')
            print(model_path, "Setting pad to special token.")
            tokenizer.add_special_tokens({'pad_token': '<unk>'})
            model.config.pad_token_id = tokenizer.pad_token_id
            tokenizer.padding_side = 'right'

    print(tokenizer.vocab_size, tokenizer.pad_token)

    return model, tokenizer

def generate_output(model, model_name, tokenizer, test_dataset, subdirectory, max_seq_length, batch_size = 8, save = False):

    model.eval()

    torch.cuda.empty_cache()
    device = model.device
    input_texts = test_dataset['prompt']

    if 'llama' in model_name.lower() or 'mistral' in model_name.lower() or 'distill' in model_name.lower(): #llama, qwen

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
        )
    
        # input_texts = [f"### Prompt: {text}\n ### Completion:" for text in input_texts]
        messages = [[{'role': 'user', 'content': text}] for text in input_texts]

        pairs = []
        set_seed(5)
        if 'distill' not in model_name:
            for generation in pipe(messages, max_new_tokens = max_seq_length, batch_size=batch_size): #temperature = 0.9, repetition_penalty = 2.0, no_repeat_ngram_size = 3, do_sample = True, top_p = 0.9,
                prompt_tmp = generation[0]['generated_text'][0]['content']
                generation_tmp = generation[0]['generated_text'][1]['content']
                pairs.append({'prompt': prompt_tmp, 'generation': generation_tmp}) 
        else:
            for generation in pipe(messages, max_new_tokens = max_seq_length, batch_size=batch_size): #temperature = 0.9, repetition_penalty = 2.0, no_repeat_ngram_size = 3, do_sample = True, top_p = 0.9,
                prompt_tmp = generation[0]['generated_text'][0]['content']
                generation_tmp = generation[0]['generated_text'][1]['content']
                pairs.append({'prompt': prompt_tmp, 'generation': generation_tmp}) 


        if save == True:
            save_generated_sentences([pair['generation'] for pair in pairs], f"outputs/generations/{subdirectory}" + model_name + ".json")

def save_generated_sentences(generations, fn):
    with open(fn, "w") as file:
        json.dump(generations, file)

################################################################################ TEXT ANALYSIS

def wcs(sentences):
    return [len(sentence.split(' ')) for sentence in sentences]

################################################################################ MINOR OUTPUT CLEANING - REMNANT OF CHAT TEMPLATE PROBABLY

def clean_up_chat_template_leftovers(s):
    if s.endswith('\n'):
        s = s.rstrip('\n')
    if s.startswith('\n\n'):
        s = s[len('\n\n'):]
    return s.strip()

################################################################################ PROMPT STRUCTURING

def select_prompt(original_sentence, source, prompt_picker, leftover_list):

    original_dev_sentence1 = 'tmp'
    simple_dev_sentence1 = 'tmp'
    original_dev_sentence2 = 'tmp'
    simple_dev_sentence2 = 'tmp'
    
    if source == 'sft':
        prompt_set_len = 10
    if source == 'dpo_pair_generation':
        prompt_set_len = 8
    if prompt_picker == 'rand':
        rand_index = random.randint(0, prompt_set_len-1)
        if rand_index >= 8:  
            dev_pair1 = leftover_list.pop(0)
            original_dev_sentence1 = dev_pair1['original']
            simple_dev_sentence1 = dev_pair1['completion']
        if rand_index == 9:
            dev_pair2 = leftover_list.pop(0)
            original_dev_sentence2 = dev_pair2['original']
            simple_dev_sentence2 = dev_pair2['completion']

    # from web app pilot, prompts not used in sft or dpo_pair_generation
    if source == 'openai':
        prompt_set = [
            f'Vereinfache den folgenden Satz auf Sprachniveau A2: {original_sentence}',
            f'Vereinfache den folgenden Satz auf Sprachniveau A2 ohne Ihre Vereinfachung zu erläutern oder voranzustellen: {original_sentence}',
            f'Schreibe den folgenden Satz in Leichter Sprache auf Sprachniveau A2: {original_sentence}',
            f'Vereinfache den folgenden Satz auf Sprachniveau A2, sodass ein Menschen mit kognitiver Behinderung ihn vertstehen würde: {original_sentence}',
            f'Schreibe den folgenden komplexen Satz um und verwende einfachere Wörter, kürzere Sätze und einfachere grammatikalische Strukturen. Die vereinfachten Texte auf Deutsch sollten dem Sprachniveau A2 entsprechen. Der Inhalt soll dabei erhalten bleiben. Komplex: {original_sentence} Leicht:',
            f'Formulieren Sie den komplexen Satz um, indem Sie mindestens einen neuen einfachen Satz bilden. Die vereinfachten Texte auf Deutsch sollten dem Sprachniveau A2 entsprechen. Behalten Sie die gleiche Bedeutung des Ausgangssatzes bei. Bitte geben Sie ausschliesslich den vereinfachten Satz an, ohne zusätzliche Informationen. Komplex: {original_sentence} Leicht:',
            f'Schreibe den folgenden komplexen Satz in Leichter Sprache. Die vereinfachten Texte auf Deutsch sollten dem Sprachniveau A2 entsprechen. Der Satz oder die Sätze Ihrer Vereinfachung sollten kurz und von geringer Komplexität sein (durchschnittlich acht bis zehn Wörter pro Satz) und eine geringe Anzahl von Aussagen pro Satz enthalten. Komplex: {original_sentence} Leicht:',
            f'Schreibe den folgenden komplexen Satz in Leichter Sprache. Die vereinfachten Texte auf Deutsch sollten dem Sprachniveau A2 entsprechen. Die Wörter in Ihrer Vereinfachung sollten kurz, beschreibend, häufig verwendet, alltagsnah und im Kontext vertraut sein. Komplex: {original_sentence} Leicht:',
            f'Schreibe den folgenden komplexen Satz in Leichter Sprache. Die vereinfachten Texte auf Deutsch sollten dem Sprachniveau A2 entsprechen. Ihre Vereinfachung sollte den zentralen Wortschatz der deutschen Sprache benutzen und für österreichische Leser:innen verständlich sein. Komplex: {original_sentence} Leicht:',
            f'Schreibe den folgenden komplexen Satz in Leichter Sprache. Die vereinfachten Texte auf Deutsch sollten dem Sprachniveau A2 entsprechen. Halten Sie sich an die Richtlinien des Buches "Leichte Sprache: das Regelbuch" von Christiane Maaß aus dem Jahr 2015. Geben Sie Ihre Vereinfachung ohne weiteren Metakommentar an. Komplex: {original_sentence} Leicht:'
        ]
    elif source == 'sft':
        prompt_set = [
            f'Schreibe den folgenden Satz in Leichter Sprache um: {original_sentence}. Bitte gib nur eine Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare.',
            f'Vereinfache den folgenden Satz, sodass Menschen mit kognitiver Beeinträchtigung den vereinfachten Satz vertstehen können: {original_sentence}. Bitte gib nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare.',
            f'Schreibe den folgenden komplexen Satz um und verwende einfachere Wörter, kürzere Sätze und reduzierte grammatikalische Strukturen. Der Inhalt und die Bedeutung sollen nach dem Umschreiben unverändert bleiben. Bitte gib nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Komplex: {original_sentence} Leicht:',
            f'Formulieren Sie den komplexen Satz um, indem Sie mindestens einen neuen einfachen Satz bilden. Behalten Sie die gleiche Bedeutung des Ausgangssatzes bei. Geben Sie bitte nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Komplex: {original_sentence} Leicht:',
            f'Schreibe den folgenden komplexen Satz in Leichter Sprache um. Die Vereinfachung soll kurz und von geringer Komplexität sein (durchschnittlich acht bis fünfzehn Wörter pro Satz) und eine geringe Anzahl von Aussagen pro Satz enthalten. Bitte gib nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Komplex: {original_sentence} Leicht:',
            f'Schreibe den folgenden komplexen Satz in Leichter Sprache um. Die Wörter in deiner Vereinfachung sollen kurz, beschreibend, und häufig verwendet von Menschen mit kognitiver Beeinträchtigung sein. Bitte gib nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Komplex: {original_sentence} Leicht:',
            f'Schreibe den folgenden komplexen Satz in Leichter Sprache um. Deine Vereinfachung soll für Menschen mit kognitiver Beeinträchtigung in Österreich verständlich sein. Bitte gib nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Komplex: {original_sentence} Leicht:',
            f'Schreiben Sie den folgenden komplexen Satz in Leichter Sprache um. Sie können 1) den Satz in mehrere Sätze aufteilen, 2) die Wortstellung ändern, um die Grammatik zu vereinfachen, 3) Wörter hinzufügen, um schwierige Konzepte zu erklären, 4) Wörter, die sich mit unnötigen Informationen zusammenhängen, entfernen, und 5) schwierige Wörter durch einfachere Vokabeln ersetzen. Achten Sie darauf, dass der Satz leichter verständlich bleibt, ohne die Bedeutung zu verändern. Bitte geben Sie nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Komplex: {original_sentence} Leicht:',
            f'Schreibe den folgenden komplexen Satz in Leichter Sprache um. Bitte gib nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Hier ist ein Beispiel. Komplex: {original_dev_sentence1} Leicht: {simple_dev_sentence1} Schreibe deine Vereinfachung nach "Leicht:". Komplex: {original_sentence} Leicht:',
            f'Schreibe den folgenden komplexen Satz in Leichter Sprache um. Bitte gib nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Hier sind zwei Beispiele. Komplex: {original_dev_sentence1} Leicht: {simple_dev_sentence1} Komplex: {original_dev_sentence2} Leicht: {simple_dev_sentence2} Schreibe deine Vereinfachung nach "Leicht:". Komplex: {original_sentence} Leicht:',
        ]

        return prompt_set[rand_index], leftover_list
    elif source == 'dpo_pair_generation':
        prompt_set = [
            f'Schreibe den folgenden Satz in Leichter Sprache um: {original_sentence}. Bitte gib nur eine Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare.',
            f'Vereinfache den folgenden Satz, sodass Menschen mit kognitiver Beeinträchtigung den vereinfachten Satz vertstehen können: {original_sentence}. Bitte gib nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare.',
            f'Schreibe den folgenden komplexen Satz um und verwende einfachere Wörter, kürzere Sätze und reduzierte grammatikalische Strukturen. Der Inhalt und die Bedeutung sollen nach dem Umschreiben unverändert bleiben. Bitte gib nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Komplex: {original_sentence} Leicht:',
            f'Formulieren Sie den komplexen Satz um, indem Sie mindestens einen neuen einfachen Satz bilden. Behalten Sie die gleiche Bedeutung des Ausgangssatzes bei. Geben Sie bitte nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Komplex: {original_sentence} Leicht:',
            f'Schreibe den folgenden komplexen Satz in Leichter Sprache um. Die Vereinfachung soll kurz und von geringer Komplexität sein (durchschnittlich acht bis fünfzehn Wörter pro Satz) und eine geringe Anzahl von Aussagen pro Satz enthalten. Bitte gib nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Komplex: {original_sentence} Leicht:',
            f'Schreibe den folgenden komplexen Satz in Leichter Sprache um. Die Wörter in deiner Vereinfachung sollen kurz, beschreibend, und häufig verwendet von Menschen mit kognitiver Beeinträchtigung sein. Bitte gib nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Komplex: {original_sentence} Leicht:',
            f'Schreibe den folgenden komplexen Satz in Leichter Sprache um. Deine Vereinfachung soll für Menschen mit kognitiver Beeinträchtigung in Österreich verständlich sein. Bitte gib nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Komplex: {original_sentence} Leicht:',
            f'Schreiben Sie den folgenden komplexen Satz in Leichter Sprache um. Sie können 1) den Satz in mehrere Sätze aufteilen, 2) die Wortstellung ändern, um die Grammatik zu vereinfachen, 3) Wörter hinzufügen, um schwierige Konzepte zu erklären, 4) Wörter, die sich mit unnötigen Informationen zusammenhängen, entfernen, und 5) schwierige Wörter durch einfachere Vokabeln ersetzen. Achten Sie darauf, dass der Satz leichter verständlich bleibt, ohne die Bedeutung zu verändern. Bitte geben Sie nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Komplex: {original_sentence} Leicht:',
        ]

        return prompt_set[rand_index]
    else:
        return prompt_set[prompt_picker]
    

def select_prompt_ATTEMPT(original_sentence, source, prompt_picker, leftover_list):

    original_dev_sentence1 = 'tmp'
    simple_dev_sentence1 = 'tmp'
    original_dev_sentence2 = 'tmp'
    simple_dev_sentence2 = 'tmp'

    if source == 'sft':
        prompt_set_len = 10
    if source == 'dpo_pair_generation':
        prompt_set_len = 8
    if prompt_picker == 'rand':
        rand_index = random.randint(0, prompt_set_len-1)
        if rand_index >= 8:  
            dev_pair1 = leftover_list.pop(0)
            original_dev_sentence1 = dev_pair1['original']
            simple_dev_sentence1 = dev_pair1['completion']
        if rand_index == 9:
            dev_pair2 = leftover_list.pop(0)
            original_dev_sentence2 = dev_pair2['original']
            simple_dev_sentence2 = dev_pair2['completion']

    # from web app pilot, prompts not used in sft or dpo_pair_generation
    if source == 'openai':
        prompt_set = [
            f'Vereinfache den folgenden Satz auf Sprachniveau A2: {original_sentence}',
            f'Vereinfache den folgenden Satz auf Sprachniveau A2 ohne Ihre Vereinfachung zu erläutern oder voranzustellen: {original_sentence}',
            f'Schreibe den folgenden Satz in Leichter Sprache auf Sprachniveau A2: {original_sentence}',
            f'Vereinfache den folgenden Satz auf Sprachniveau A2, sodass ein Menschen mit kognitiver Behinderung ihn vertstehen würde: {original_sentence}',
            f'Schreibe den folgenden komplexen Satz um und verwende einfachere Wörter, kürzere Sätze und einfachere grammatikalische Strukturen. Die vereinfachten Texte auf Deutsch sollten dem Sprachniveau A2 entsprechen. Der Inhalt soll dabei erhalten bleiben. Komplex: {original_sentence} Leicht:',
            f'Formulieren Sie den komplexen Satz um, indem Sie mindestens einen neuen einfachen Satz bilden. Die vereinfachten Texte auf Deutsch sollten dem Sprachniveau A2 entsprechen. Behalten Sie die gleiche Bedeutung des Ausgangssatzes bei. Bitte geben Sie ausschliesslich den vereinfachten Satz an, ohne zusätzliche Informationen. Komplex: {original_sentence} Leicht:',
            f'Schreibe den folgenden komplexen Satz in Leichter Sprache. Die vereinfachten Texte auf Deutsch sollten dem Sprachniveau A2 entsprechen. Der Satz oder die Sätze Ihrer Vereinfachung sollten kurz und von geringer Komplexität sein (durchschnittlich acht bis zehn Wörter pro Satz) und eine geringe Anzahl von Aussagen pro Satz enthalten. Komplex: {original_sentence} Leicht:',
            f'Schreibe den folgenden komplexen Satz in Leichter Sprache. Die vereinfachten Texte auf Deutsch sollten dem Sprachniveau A2 entsprechen. Die Wörter in Ihrer Vereinfachung sollten kurz, beschreibend, häufig verwendet, alltagsnah und im Kontext vertraut sein. Komplex: {original_sentence} Leicht:',
            f'Schreibe den folgenden komplexen Satz in Leichter Sprache. Die vereinfachten Texte auf Deutsch sollten dem Sprachniveau A2 entsprechen. Ihre Vereinfachung sollte den zentralen Wortschatz der deutschen Sprache benutzen und für österreichische Leser:innen verständlich sein. Komplex: {original_sentence} Leicht:',
            f'Schreibe den folgenden komplexen Satz in Leichter Sprache. Die vereinfachten Texte auf Deutsch sollten dem Sprachniveau A2 entsprechen. Halten Sie sich an die Richtlinien des Buches "Leichte Sprache: das Regelbuch" von Christiane Maaß aus dem Jahr 2015. Geben Sie Ihre Vereinfachung ohne weiteren Metakommentar an. Komplex: {original_sentence} Leicht:'
        ]
    elif source == 'sft':
        prompt_set = [
            f'Schreibe den folgenden Satz in Leichter Sprache um: {original_sentence}. Bitte gib nur eine Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Sie müssen Wörter im Originalsatz hinzufügen, entfernen, neu anordnen oder ändern, um den Kerninhalt des Satzes in irgendeiner Weise verständlicher zu machen.',
            f'Vereinfache den folgenden Satz, sodass Menschen mit kognitiver Beeinträchtigung den vereinfachten Satz vertstehen können: {original_sentence}. Bitte gib nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Sie müssen Wörter im Originalsatz hinzufügen, entfernen, neu anordnen oder ändern, um den Kerninhalt des Satzes in irgendeiner Weise verständlicher zu machen.',
            f'Schreibe den folgenden komplexen Satz um und verwende einfachere Wörter, kürzere Sätze und reduzierte grammatikalische Strukturen. Der Inhalt und die Bedeutung sollen nach dem Umschreiben unverändert bleiben. Bitte gib nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Sie müssen Wörter im Originalsatz hinzufügen, entfernen, neu anordnen oder ändern, um den Kerninhalt des Satzes in irgendeiner Weise verständlicher zu machen. Komplex: {original_sentence} Leicht:',
            f'Formulieren Sie den komplexen Satz um, indem Sie mindestens einen neuen einfachen Satz bilden. Behalten Sie die gleiche Bedeutung des Ausgangssatzes bei. Geben Sie bitte nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Sie müssen Wörter im Originalsatz hinzufügen, entfernen, neu anordnen oder ändern, um den Kerninhalt des Satzes in irgendeiner Weise verständlicher zu machen. Komplex: {original_sentence} Leicht:',
            f'Schreibe den folgenden komplexen Satz in Leichter Sprache um. Die Vereinfachung soll kurz und von geringer Komplexität sein (durchschnittlich acht bis fünfzehn Wörter pro Satz) und eine geringe Anzahl von Aussagen pro Satz enthalten. Bitte gib nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Sie müssen Wörter im Originalsatz hinzufügen, entfernen, neu anordnen oder ändern, um den Kerninhalt des Satzes in irgendeiner Weise verständlicher zu machen. Komplex: {original_sentence} Leicht:',
            f'Schreibe den folgenden komplexen Satz in Leichter Sprache um. Die Wörter in deiner Vereinfachung sollen kurz, beschreibend, und häufig verwendet von Menschen mit kognitiver Beeinträchtigung sein. Bitte gib nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Sie müssen Wörter im Originalsatz hinzufügen, entfernen, neu anordnen oder ändern, um den Kerninhalt des Satzes in irgendeiner Weise verständlicher zu machen. Komplex: {original_sentence} Leicht:',
            f'Schreibe den folgenden komplexen Satz in Leichter Sprache um. Deine Vereinfachung soll für Menschen mit kognitiver Beeinträchtigung in Österreich verständlich sein. Bitte gib nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Sie müssen Wörter im Originalsatz hinzufügen, entfernen, neu anordnen oder ändern, um den Kerninhalt des Satzes in irgendeiner Weise verständlicher zu machen. Komplex: {original_sentence} Leicht:',
            f'Schreiben Sie den folgenden komplexen Satz in Leichter Sprache um. Sie können 1) den Satz in mehrere Sätze aufteilen, 2) die Wortstellung ändern, um die Grammatik zu vereinfachen, 3) Wörter hinzufügen, um schwierige Konzepte zu erklären, 4) Wörter, die sich mit unnötigen Informationen zusammenhängen, entfernen, und 5) schwierige Wörter durch einfachere Vokabeln ersetzen. Achten Sie darauf, dass der Satz leichter verständlich bleibt, ohne die Bedeutung zu verändern. Bitte geben Sie nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Sie müssen Wörter im Originalsatz hinzufügen, entfernen, neu anordnen oder ändern, um den Kerninhalt des Satzes in irgendeiner Weise verständlicher zu machen. Komplex: {original_sentence} Leicht:',
            f'Schreibe den folgenden komplexen Satz in Leichter Sprache um. Bitte gib nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Sie müssen Wörter im Originalsatz hinzufügen, entfernen, neu anordnen oder ändern, um den Kerninhalt des Satzes in irgendeiner Weise verständlicher zu machen. Komplex: {original_dev_sentence1} Leicht: {simple_dev_sentence1} Schreibe deine Vereinfachung nach "Leicht:". Komplex: {original_sentence} Leicht:',
            f'Schreibe den folgenden komplexen Satz in Leichter Sprache um. Bitte gib nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Sie müssen Wörter im Originalsatz hinzufügen, entfernen, neu anordnen oder ändern, um den Kerninhalt des Satzes in irgendeiner Weise verständlicher zu machen. Komplex: {original_dev_sentence1} Leicht: {simple_dev_sentence1} Komplex: {original_dev_sentence2} Leicht: {simple_dev_sentence2} Schreibe deine Vereinfachung nach "Leicht:". Komplex: {original_sentence} Leicht:',
        ]

        return prompt_set[rand_index], leftover_list
    elif source == 'dpo_pair_generation':
        prompt_set = [
            f'Schreibe den folgenden Satz in Leichter Sprache um: {original_sentence}. Bitte gib nur eine Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare.',
            f'Vereinfache den folgenden Satz, sodass Menschen mit kognitiver Beeinträchtigung den vereinfachten Satz vertstehen können: {original_sentence}. Bitte gib nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare.',
            f'Schreibe den folgenden komplexen Satz um und verwende einfachere Wörter, kürzere Sätze und reduzierte grammatikalische Strukturen. Der Inhalt und die Bedeutung sollen nach dem Umschreiben unverändert bleiben. Bitte gib nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Komplex: {original_sentence} Leicht:',
            f'Formulieren Sie den komplexen Satz um, indem Sie mindestens einen neuen einfachen Satz bilden. Behalten Sie die gleiche Bedeutung des Ausgangssatzes bei. Geben Sie bitte nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Komplex: {original_sentence} Leicht:',
            f'Schreibe den folgenden komplexen Satz in Leichter Sprache um. Die Vereinfachung soll kurz und von geringer Komplexität sein (durchschnittlich acht bis fünfzehn Wörter pro Satz) und eine geringe Anzahl von Aussagen pro Satz enthalten. Bitte gib nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Komplex: {original_sentence} Leicht:',
            f'Schreibe den folgenden komplexen Satz in Leichter Sprache um. Die Wörter in deiner Vereinfachung sollen kurz, beschreibend, und häufig verwendet von Menschen mit kognitiver Beeinträchtigung sein. Bitte gib nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Komplex: {original_sentence} Leicht:',
            f'Schreibe den folgenden komplexen Satz in Leichter Sprache um. Deine Vereinfachung soll für Menschen mit kognitiver Beeinträchtigung in Österreich verständlich sein. Bitte gib nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Komplex: {original_sentence} Leicht:',
            f'Schreiben Sie den folgenden komplexen Satz in Leichter Sprache um. Sie können 1) den Satz in mehrere Sätze aufteilen, 2) die Wortstellung ändern, um die Grammatik zu vereinfachen, 3) Wörter hinzufügen, um schwierige Konzepte zu erklären, 4) Wörter, die sich mit unnötigen Informationen zusammenhängen, entfernen, und 5) schwierige Wörter durch einfachere Vokabeln ersetzen. Achten Sie darauf, dass der Satz leichter verständlich bleibt, ohne die Bedeutung zu verändern. Bitte geben Sie nur die Vereinfachung an, ohne Einleitung, Alternativen oder Kommentare. Komplex: {original_sentence} Leicht:',
        ]

        return prompt_set[rand_index]
    else:
        return prompt_set[prompt_picker]
    
################################################################################ VISUALIZATION

def return_proper_model_name(model_name):
    if 'disco' in model_name:
        return 'DiscoLeo-Llama-3-8B-Instruct'
    if 'llama' in model_name and 'disco' not in model_name:
        return 'Llama-3.1-8B-Instruct'
    if 'leolm' in model_name:
        return 'LeoLM-Mistral-7B-Chat'
    if 'mistral' in model_name and 'leolm' not in model_name:
        return 'Mistral-7B-Instruct'
def return_model_color(model_name):
    if 'disco' in model_name:
        return sns.color_palette("colorblind", 11)[4] # '#CB78BC'
    if 'llama' in model_name and 'disco' not in model_name:
        return sns.color_palette("colorblind", 11)[0] #'#0273B2'
    if 'leolm' in model_name:
        return sns.color_palette("colorblind", 11)[1] #'#DE8F05'
    if 'mistral' in model_name and 'leolm' not in model_name:
        return sns.color_palette("colorblind", 11)[3] #'#D55E00'

# outdated
anonymizer_mapper = {
    'ea01': 'ex40',
    'ea02': 'ex30',
    'ea03': 'ex20',
    'ea04': 'ex10',
    'ta01': 'tg90',
    'ta02': 'tg85',
    'ta03': 'tg80',
    'ta04': 'tg75',
    'ta05': 'tg70',
    'ta06': 'tg65',
    'ta07': 'tg60',
    'ta08': 'tg55',
    'ta09': 'tg50',
    'ta10': 'tg40',
    'ta11': 'tg30',
    'ta12': 'tg25',
    'ta13': 'tg20',
    'ta14': 'tg15',
    'ta15': 'tg10'
}
def anonymizer(code_userID):
    return anonymizer_mapper[code_userID]