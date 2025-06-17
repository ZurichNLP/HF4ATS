import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import json
from transformers import pipeline
import tqdm 
import difflib
from openai import OpenAI
import pandas as pd
import numpy as np
import os
from utils import base_dependencies
import re
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
from datetime import datetime
import langdetect
from langdetect import DetectorFactory
from collections import Counter
import copy
from sentence_transformers import SentenceTransformer
from itertools import combinations
from sklearn.metrics import cohen_kappa_score  
from trl import DPOConfig, DPOTrainer
from accelerate import Accelerator
import shutil
from peft import get_peft_model, LoraConfig, PeftModel, PeftConfig, AutoPeftModelForCausalLM
import wandb
from itertools import product

# source: https://github.com/ahmed-alllam/Direct-Preference-Optimization

################################################################################ PRE-INFERENCE: PREPARE SENTENCES

def language_level_transition(s):
    parts = s.split('.')
    part1, part2, part3 = parts[0], parts[1], parts[2]
    
    LC1 = 'OR' if 'de' == part2[0:2] else part1.split('_')[-1] if 'simpde' == part2[0:6] else None
    LC2 = 'OR' if 'de' == part3[0:2] else part2.split('_')[-1] if 'simpde' == part3[0:6] else None

    return f"{LC1}-{LC2}"

def clean_sentences(sentences):
    cleaned = []
    for s in sentences:
        if not re.match(r'^[^\w]', s) and (s.strip().endswith(('.', '?', '!', "'", '"'))):
            s = re.sub(r'\s+([.,!?):])', r'\1', s)
            s = re.sub(r'\s{2,}', ' ', s)
            s = re.sub(r'\(\s+', '(', s)
            cleaned.append(s)
    return cleaned

def scrub_deplain(capito_sentences):
    deplain_all = list((set(pd.read_csv('data/deplain_sentences/all.csv')['original'].str.lower())))
    deplain_all = [re.sub(r'[^a-zA-Z]', '', sent) for sent in deplain_all]
    return [sent for sent in capito_sentences if re.sub(r'[^a-zA-Z]', '', sent.lower()) not in deplain_all]

def plt_histogram(wcs, group):
    plt.figure()
    plt.hist(wcs, bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50], edgecolor='black', align='left')
    plt.title(group + 'Word Count Distribution')
    plt.show()

def capito_aggregate():
    capito_filepath = "data/capito_sentences/"
    combined_df = pd.DataFrame(columns=['original', 'target', 'source_file'])
    for filename in os.listdir(capito_filepath):
        file_path = os.path.join(capito_filepath, filename)
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path, delimiter=";")
            result = df[df['original'] != df['target']][['original', 'target']]
            result['source_file'] = filename
            combined_df = pd.concat([combined_df, result], ignore_index=True)
    combined_df = combined_df[~combined_df['original'].isnull()]
    combined_df = combined_df[~combined_df['target'].isnull()]
    combined_df = combined_df.reset_index(drop=True)
    combined_df = combined_df[~combined_df['original'].str.contains('NULL')]
    combined_df['language_transition'] = combined_df['source_file'].apply(language_level_transition)

    from collections import Counter
    print(Counter(combined_df['language_transition']))

    print(len(combined_df), 'capito sentences.')
    print('filgering based on A2 origin...')
    combined_df = combined_df[combined_df['language_transition'] != 'A2-A1']
    print(len(combined_df), 'capito sentences.')

    capito_sentences = combined_df['original']
    capito_sentences = list(set(capito_sentences))
    capito_sentences = clean_sentences(capito_sentences)

    return combined_df

def lhapa_aggregate():

    lhapa1_filepath = "data/lhapa_sentences/B1-OR/"
    lhapa2_filepath = "data/lhapa_sentences/A2-OR/"
    lhapa_sentences = []

    for filename in os.listdir(lhapa1_filepath):
        file_path = os.path.join(lhapa1_filepath, filename)
        if os.path.isfile(file_path):
            if 'simpde' not in file_path:
                tmp = []
                with open(file_path) as file:
                    content = file.read()
                tmp = content.split('\n')
            lhapa_sentences = lhapa_sentences + tmp

    for filename in os.listdir(lhapa2_filepath):
        file_path = os.path.join(lhapa2_filepath, filename)
        if os.path.isfile(file_path):
            if 'simpde' not in file_path:
                tmp = []
                with open(file_path) as file:
                    content = file.read()
                tmp = content.split('\n')
            lhapa_sentences = lhapa_sentences + tmp
    
    lhapa_sentences = list(set(lhapa_sentences))
    lhapa_sentences = clean_sentences(lhapa_sentences)

    return lhapa_sentences

def aggregate_dpo_possibilities():
    
    #capito_sentences = capito_aggregate()
    
    lhapa_sentences = lhapa_aggregate()

    apa_sentences =  [instance_dict['original'] for instance_dict in base_dependencies.load_jsonl('data/sft_leftovers.jsonl')]
    apa_sentences = list(set(apa_sentences))
    apa_sentences = clean_sentences(apa_sentences)

    print(len(apa_sentences), 'apa sentences,', len(lhapa_sentences), 'lhapa sentences.')
    print('filtering based on in apa...')
    lhapa_sentences = scrub_deplain(lhapa_sentences)
    print(len(apa_sentences), 'apa sentences,', len(lhapa_sentences), 'lhapa sentences.')
    print('filtering based on length...')
    DetectorFactory.seed = 5

    lhapa_sentences = [sent for sent in lhapa_sentences if len(sent.split(' ')) < 51 and len(sent.split(' ')) > 4]
    print(len(apa_sentences), 'apa sentences,', len(lhapa_sentences), 'lhapa sentences.')
    print('filtering based on non-German...')
    lhapa_sentences = [sent for sent in lhapa_sentences if sent != 'Security personnel inspect the interior of St. Sebastian s Church in Negombo on April 22, 2019, a day after the church was hit in series of bomb blasts targeting churches and luxury hotels in Sri Lanka.']
    print(len(apa_sentences), 'apa sentences,', len(lhapa_sentences), 'lhapa sentences.')
    print('filtering based on multiple sentences...')
    apa_sentences = [sent for sent in apa_sentences if not re.search(r'\w[.!?](?=.{2,}$)', sent) and not re.search(r'\)[.!?](?=.{2,}$)', sent)]
    lhapa_sentences = [sent for sent in lhapa_sentences if not re.search(r'\w[.!?](?=.{2,}$)', sent) and not re.search(r'\)[.!?](?=.{2,}$)', sent)]
    print(len(apa_sentences), 'apa sentences,', len(lhapa_sentences), 'lhapa sentences.')


    id_df = pd.DataFrame(apa_sentences + lhapa_sentences)
    id_df['id'] = ['uid_apa_' + str(i) if i < len(apa_sentences) else 'uid_lhapa_' + str(i-len(apa_sentences)) for i in id_df.index]
    id_df = id_df.rename(columns={0:'original'})

    # bring in ids from pilot
    pilot_uids = pd.read_csv('outputs/dpo/pilot_original_sentence_uid.csv')
    pilot_uids = pilot_uids.rename(columns={'id':'pilot_id'})
    pilot_uids = pilot_uids[pilot_uids['pilot_id'] != 'pilot_capito_31716'] # this sentence appeared in both deplain and capito; it needs to be removed because the earlier filter didn't apply to this file.

    pd.merge(id_df, pilot_uids, on='original', how='outer').to_csv('outputs/dpo/original_sentence_uid.csv', index=False)

    return apa_sentences + lhapa_sentences

def access_sentences_and_wcs():
    base_sentence_set = pd.read_csv('outputs/dpo/original_sentence_uid.csv')
    apa_sentences = list(base_sentence_set[base_sentence_set['id'].str.contains('_apa', na=False)]['original'])
    lhapa_sentences = list(base_sentence_set[base_sentence_set['id'].str.contains('_lhapa', na=False)]['original'])
    lhapa_sentences_wcs = base_dependencies.wcs(lhapa_sentences)
    apa_sentences_wcs = base_dependencies.wcs(apa_sentences)
    return apa_sentences, apa_sentences_wcs, lhapa_sentences, lhapa_sentences_wcs

def plot_dpo_possibility_dist():

    _, apa_sentences_wcs, _, lhapa_sentences_wcs = access_sentences_and_wcs()

    print('len apa: ', len(apa_sentences_wcs))
    plt_histogram(apa_sentences_wcs, 'APA')
    print('len lhapa: ', len(lhapa_sentences_wcs))
    plt_histogram(lhapa_sentences_wcs, 'LHAPA')

    
def draw_from_sentences(sent_list, sent_list_wcs, num_samples, source, seed, train_orig_wcs_mean, share = 0, apa_wcs_samp_mean = 0):

    np.random.seed(seed)
    if source == 'APA':
        sent_list_wcs = [np.exp(-((wc - train_orig_wcs_mean) ** 2) / (2 * 3 ** 2)) for wc in sent_list_wcs]
    if source == 'LHAPA':
        sent_list_wcs = [np.exp(-((wc - (train_orig_wcs_mean + share*(train_orig_wcs_mean - apa_wcs_samp_mean))) ** 2) / (2 * 3 ** 2)) for wc in sent_list_wcs]
        #sent_list_wcs = [1/(np.abs(wc-(train_orig_wcs_mean + share_apa*train_leftover_diff)))**(1-share_apa) for wc in sent_list_wcs]
    sent_list_wcs_word_count = sum(sent_list_wcs)
    sentences_dpo_origs = np.random.choice(sent_list, size=num_samples, replace=False, p=[num / sent_list_wcs_word_count for num in sent_list_wcs])
    return sentences_dpo_origs

def draw_to_dpo_sample(seed, num, share):

    apa_amount = num*share
    cap_amount = num-apa_amount
    apa_sentences, apa_sentences_wcs, lhapa_sentences, lhapa_sentences_wcs = access_sentences_and_wcs()

    train_orig_sentences = [instance_dict['original'] for instance_dict in base_dependencies.load_jsonl('data/sft_train.jsonl')]
    train_orig_wcs = base_dependencies.wcs(train_orig_sentences)
    train_orig_wcs_mean = np.mean(train_orig_wcs)
    apa_sentences_wcs_mean = np.mean(apa_sentences_wcs)
    
    # cutoff higher values because they don't match lhapa distribution
    lhapa_zip = zip(lhapa_sentences, lhapa_sentences_wcs)
    lhapa_30orless = [sent_tup for sent_tup in lhapa_zip if sent_tup[1] <= 30]
    lhapa_sentences = [sent_tup[0] for sent_tup in lhapa_30orless]
    lhapa_sentences_wcs = [sent_tup[1] for sent_tup in lhapa_30orless]

    apa_subset = draw_from_sentences(apa_sentences, apa_sentences_wcs, int(apa_amount), "APA", seed, train_orig_wcs_mean)
    lhapa_subset = draw_from_sentences(lhapa_sentences, lhapa_sentences_wcs, int(cap_amount), "LHAPA", seed, train_orig_wcs_mean, share, np.mean(base_dependencies.wcs(apa_subset)))
    dpo_sentences = np.concatenate((apa_subset, lhapa_subset))
    plt_histogram(base_dependencies.wcs(dpo_sentences), "Sample")
    dpo_sentences = pd.DataFrame(dpo_sentences)
    dpo_sentences = dpo_sentences.rename(columns={0 :'original'})
    return dpo_sentences, apa_subset, lhapa_subset

################################################################################ LOSS MECHANICS FOR INFERENCE

def get_log_probs(logits, labels, tokenizer):

    log_probs = F.log_softmax(logits, dim=-1)

    ## Get the top 3 token IDs based on log probabilities
    #top_k = 3
    #top_k_values, top_k_token_ids = torch.topk(log_probs, top_k, dim=-1)  # Get top 3 values and indices for each timestep
    #top_k_values = top_k_values.squeeze(0)
    #top_k_token_ids = top_k_token_ids.squeeze(0)
    ## Display the top 3 token IDs and their corresponding decoded tokens
    #for idx in range(top_k_token_ids.size(0)):  # Iterate through each timestep
    #    print(f"Step {idx+1}:")
    #    for rank in range(top_k_token_ids[idx].size(0)):  # Iterate over the top k tokens for this timestep
    #        token_id = top_k_token_ids[idx][rank]
    #        log_prob = top_k_values[idx][rank]
    #        # Decode each token_id individually
    #        decoded_token = tokenizer.decode([token_id.item()])  # Decode each token_id
    #        print(f"  Rank {rank+1}: Token ID: {token_id.item()} -> Token: {decoded_token} | Log Prob: {log_prob.item()}")

    #print(logits.shape)
    #print(log_probs.shape)
    #print(labels.shape)
    #for idx in range(logits.size(1)):
    #    print(idx, log_probs[0, idx, 39525])
    
    return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)

def inference_log_probs(model, tokenizer, prompt, simplification, max_seq_length):

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    simp_ids = tokenizer(simplification, return_tensors="pt", max_length = max_seq_length, truncation = True, add_special_tokens = False).input_ids.to(device)
    concat_ids = torch.cat([prompt_ids, simp_ids], dim=-1).to(device)
    
    #for token_id in concat_ids.squeeze(0):
    #    decoded_token = tokenizer.decode([token_id.item()])  # Decode each token_id
    #    print(f"Token ID: {token_id.item()} -> Token: {decoded_token}")

    with torch.no_grad():
        logits = model(concat_ids).logits
    
    #print(prompt_ids.size(1))

    simplification_logits = logits[:, prompt_ids.size(1)-1:-1, :] 
    simplification_labels = simp_ids 
    
    #print(simplification_logits)
    #for token in simplification_labels[0]:
    #    print(token, tokenizer.decode(token, skip_special_tokens=False))

    return get_log_probs(simplification_logits, simplification_labels, tokenizer)

################################################################################ DPO INFERENCE

def pair_generation_sampling_sft(model, model_name, tokenizer, inference_sentences, max_seq_length, p, t, ns, pre_dpo = False):

    print(f"Pairs from {model_name}, sampling:")

    torch.cuda.empty_cache()

    print(model_name)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
    )
    
    chunk_size = 400
    total_chunks = len(inference_sentences) // chunk_size + (len(inference_sentences) % chunk_size > 0)

    #stemodel = base_dependencies.launch_embedding_model()
    #translation_model, translation_processor = base_dependencies.launch_translation_model()

    for chunk_index in tqdm.tqdm(range(total_chunks), desc=f"inference, {chunk_size} obs in a chunk"):
        start = chunk_index * chunk_size
        end = min((chunk_index + 1) * chunk_size, len(inference_sentences))
        chunk = inference_sentences[start:end]

        partial_fname = model_name.replace('/', '_') + f'_p{p}t{t}_' + str(start) + '_' + str(end)
        print(partial_fname)
        if not any(partial_fname in filename for filename in os.listdir('outputs/dpo/post_inference_sampling/')):
            print('generating up to obs', end)
            gen_dicts = gen_via_pipe(pipe, chunk, max_seq_length, p, t, ns)

            # deduplicate generations here to save time on forward pass and embeddings
            # THIS IS WHY WE SEE LESS THAN 10 GENERATIONS DESPITE ASKING FOR 10 IN GEN_VIA_PIPE()
            for gen_dict in gen_dicts:
                gen_dict['generations'] = [base_dependencies.clean_up_chat_template_leftovers(gen) for gen in list(set(gen_dict['generations'])) if len(base_dependencies.clean_up_chat_template_leftovers(gen)) > 0]

            # add in all filtering info
            for gen_dict in gen_dicts:
                #gen_dict['embeddings'] = [stemodel.encode(gen) for gen in gen_dict['generations']]
                gen_dict['logprobs'] = [inference_log_probs(model, tokenizer, gen_dict['prompt'], gen, max_seq_length=max_seq_length).squeeze().tolist() for gen in gen_dict['generations']]
                #gen_dict['simp_translations'] = base_dependencies.translate_german_sentences(translation_model, translation_processor, gen_dict['generations'], len(gen_dict['generations']))
            
            if pre_dpo == True:
                pd.DataFrame(gen_dicts).to_json("outputs/dpo/inference_sampling/" + partial_fname + '_' + datetime.now().strftime('%Y%m%d_%H%M') + ".jsonl", orient='records', lines=True)

            if pre_dpo == False:
                pd.DataFrame(gen_dicts).to_json("outputs/dpo/post_inference_sampling/" + partial_fname + '_' + datetime.now().strftime('%Y%m%d_%H%M') + ".jsonl", orient='records', lines=True)

def gen_via_pipe(pipe, chunk, max_seq_length, p, t, ns):
    gens = []
    input_texts = chunk['prompt']
    messages = [[{'role': 'user', 'content': text}] for text in input_texts]
    for generation in pipe(messages, max_new_tokens = max_seq_length, batch_size=20, num_return_sequences=ns, do_sample=True, top_p = p, temperature = t):
        prompt_tmp = generation[0]['generated_text'][0]['content']
        content = [[inner['content'] for inner in d['generated_text']] for d in generation]
        generations_tmp = [dialogue[1] for dialogue in content]
        gens.append({'prompt': prompt_tmp, 'generations': generations_tmp})
    i = 0
    for gen_dict in gens:
        gen_dict['original'] = chunk.iloc[i]['original']
        #gen_dict['original_english'] = chunk.iloc[i]['original_english']
        i += 1
    return gens

def logprobs_via_forward(model, tokenizer, gen_dicts, max_seq_length):

    gen_logprobs_dict = {}

    # set of unique prompt-generation pairs
    gens_unique = set()
    for gen_dict in gen_dicts:
        prompt = gen_dict['prompt']
        for gen in gen_dict['generations']:
            gens_unique.add((prompt, gen))

    # log_probs for the unique pairs found above
    for prompt_gen_tuple in gens_unique:
        log_probs = inference_log_probs(model, tokenizer, prompt_gen_tuple[0], prompt_gen_tuple[1], max_seq_length=max_seq_length)
        gen_logprobs_dict[(prompt_gen_tuple[0], prompt_gen_tuple[1])] = log_probs.squeeze().tolist()

    # throw logprobs into final output
    for gen_dict in gen_dicts:
        gen_dict['logprobs'] = [gen_logprobs_dict[(gen_dict['prompt'], gen_dict_gen)] for gen_dict_gen in gen_dict['generations']]
    
    return gen_dicts

def api_call_sampling_openai(prompt, openai_model = 'gpt-4o'):

    client = OpenAI(api_key='<YOUR_OPENAI_API_KEY>')  # RESEARCHER MUST INPUT KEY

    completion = client.chat.completions.create(
        model=openai_model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        top_p=0.9,
        frequency_penalty=0.75,
        logprobs=True,
        n=5
    )

    generations = []
    for response in completion.choices:
        generation = response.message.content
        perplexity = np.exp(-np.mean([token.logprob for token in response.logprobs.content]))
        generations.append((generation, perplexity))

    return generations
    
def pair_generation_sampling_openai(prompt_dict_set, save_name):
    
    for dict in prompt_dict_set:
        dict['generations'] = api_call_sampling_openai(dict['prompt'])    

    pd.DataFrame(prompt_dict_set).to_json(f'outputs/dpo/inference_sampling/' + save_name, orient='records', lines=True)

################################################################################ POST INFERENCE - ADD FILTERING INFORMATION

def aggregate_dpo_inference_simplifications():

    raw_inference_directory = "outputs/dpo/inference_sampling/"
    models = ['sft_disco_llama8b_checkpoint2800', 'sft_llama8b_checkpoint2400', 'sft_leolm_mistral7b_checkpoint1600']
    simplification_set = []

    # aggregate possibilities + associated perplexities
    for filename in os.listdir(raw_inference_directory):
        for model in models:
            simplifications_model = []
            if model in filename and filename.startswith('bs16ga1dv1lr1e-4') and filename.endswith('.jsonl'):
                file_path = os.path.join(raw_inference_directory, filename)
                with open(file_path, "r") as file:
                    for line in file:
                        simplifications_model.append(json.loads(line))
            for simp_dict in simplifications_model:
                simp_dict['generations'] = [(generation, np.exp(-np.mean(log_probs))) for generation, log_probs in zip(simp_dict['generations'], simp_dict['logprobs'])]
                simp_dict['model'] = model
                simp_dict['filename'] = filename
                del simp_dict['logprobs']

            simplification_set += simplifications_model

    simplification_set = pd.DataFrame(simplification_set).explode('generations', ignore_index=True)
    simplification_set[['simplification', 'perplexity']] = pd.DataFrame(
        simplification_set['generations'].tolist(), index=simplification_set.index
    )
    simplification_set = simplification_set.drop(columns=['generations'])

    # attach original sentence ids
    orig_id = pd.read_csv('outputs/dpo/original_sentence_uid.csv')
    orig_id = orig_id[~orig_id['id'].isna()]
    orig_id = orig_id.drop(columns={'pilot_id'})
    orig_id = orig_id.rename(columns={'id':'orig_id'})
    simplification_set = pd.merge(simplification_set, orig_id, how='left', on='original', validate='m:1')
    simplification_set = simplification_set[['filename', 'model', 'orig_id', 'prompt', 'original', 'simplification', 'perplexity']]
    simplification_set = simplification_set[simplification_set['original'] != simplification_set['simplification']]

    return simplification_set

def translate_sentences(df_w_sentences, col_to_translate):

    translated_col = f'{col_to_translate}_english'

    # sentences with preexisting translations
    cumulative_store_de_en_list = base_dependencies.load_jsonl(f'outputs/dpo/cumulative_stores/{col_to_translate}_de_en.jsonl')
    previous_translations = pd.DataFrame(cumulative_store_de_en_list)

    # filter down to sentence without prexisting translations

    untranslated_subset = df_w_sentences[~df_w_sentences[col_to_translate].isin(previous_translations[col_to_translate])]
    untranslated_subset = untranslated_subset.drop_duplicates(subset=[col_to_translate], keep='first')

    # if any sentences for dpo haven't been translated, translate them. otherwise, proceed to inference.
    if len(untranslated_subset) > 0:
        print(f'translating {len(untranslated_subset)} {col_to_translate} to english')
        model, processor = base_dependencies.launch_translation_model()
        untranslated_subset[f'{col_to_translate}_english'] = base_dependencies.translate_german_sentences(model, processor, list(untranslated_subset[col_to_translate]), 100)
        new_cum_store = pd.concat([previous_translations, untranslated_subset[[col_to_translate, translated_col]]], ignore_index=True).reset_index(drop=True)
        new_cum_store.to_json(f'outputs/dpo/cumulative_stores/{col_to_translate}_de_en.jsonl', orient='records', lines=True)
    else: 
        pass

    previous_translations = pd.DataFrame(base_dependencies.load_jsonl(f'outputs/dpo/cumulative_stores/{col_to_translate}_de_en.jsonl'))
    df_w_sentences = pd.merge(df_w_sentences, previous_translations, how='left', on=[col_to_translate], validate='m:1')
    return df_w_sentences

def cossim_sentences(df_w_sentences, col1, col2):

    cossim_col = f'{col1}_{col2}_cossim'
    # col1->col2 pairs with preexisting cossims
    cumulative_store_cossim_list = base_dependencies.load_jsonl(f'outputs/dpo/cumulative_stores/{col1}_{col2}_cossim.jsonl')
    previous_cossims = pd.DataFrame(cumulative_store_cossim_list)

    # filter down to pairs without prexisting cossims
    uncossim_subset = df_w_sentences[~(df_w_sentences['original'].isin(previous_cossims['original']) & 
                                    df_w_sentences['simplification'].isin(previous_cossims['simplification']))]
    uncossim_subset = uncossim_subset.drop_duplicates(subset=[col1, col2], keep='first')

    # if any sentence pairs for dpo haven't been cossimed, cossim them. otherwise, proceed to inference.
    if len(uncossim_subset) > 0:
        print(f'calculating cossim for {len(uncossim_subset)} {col1} {col2} pairs')
        stemodel = base_dependencies.launch_embedding_model()
        uncossim_subset[cossim_col] = base_dependencies.cossim_sentence_lists(stemodel, list(uncossim_subset[col1]), list(uncossim_subset[col2]))
        new_cum_store = pd.concat([previous_cossims, uncossim_subset[[col1, col2, cossim_col]]], ignore_index=True).reset_index(drop=True)
        new_cum_store.to_json(f'outputs/dpo/cumulative_stores/{col1}_{col2}_cossim.jsonl', orient='records', lines=True)
    else: 
        pass

    previous_cossims = pd.DataFrame(base_dependencies.load_jsonl(f'outputs/dpo/cumulative_stores/{col1}_{col2}_cossim.jsonl'))
    df_w_sentences = pd.merge(df_w_sentences, previous_cossims, how='left', on=[col1, col2], validate='m:1')
    return df_w_sentences

def entailment_inference(df_w_sentences, col1, col2):

    entail_col = f'{col1}_{col2}_entailment'

    cumulative_store_entail_list = base_dependencies.load_jsonl(f'outputs/dpo/cumulative_stores/{entail_col}.jsonl')
    previous_entails = pd.DataFrame(cumulative_store_entail_list)

    unentail_subset = df_w_sentences[~(df_w_sentences['original'].isin(previous_entails['original']) & 
                                    df_w_sentences['simplification'].isin(previous_entails['simplification']))]
    unentail_subset = unentail_subset.drop_duplicates(subset=[col1, col2], keep='first')

    # if any sentence pairs for dpo haven't been entailed, entail them. otherwise, proceed to inference.
    if len(unentail_subset) > 0:
        print(f'calculating entail for {len(unentail_subset)} {col1} {col2} pairs')
        entail_model, entail_tokenizer = base_dependencies.launch_entailment_model()
        unentail_subset[entail_col] = base_dependencies.entail_sentence_lists(entail_model, entail_tokenizer, list(unentail_subset[col1]), list(unentail_subset[col2]))
        new_cum_store = pd.concat([previous_entails, unentail_subset[[col1, col2, entail_col]]], ignore_index=True).reset_index(drop=True)
        new_cum_store.to_json(f'outputs/dpo/cumulative_stores/{entail_col}.jsonl', orient='records', lines=True)
    else: 
        pass

    previous_entails = pd.DataFrame(base_dependencies.load_jsonl(f'outputs/dpo/cumulative_stores/{entail_col}.jsonl'))
    df_w_sentences = pd.merge(df_w_sentences, previous_entails, how='left', on=[col1, col2], validate='m:1')
    return df_w_sentences

def complexity_sentences(df_w_sentences, col_to_complex):

    complexity_col = f'{col_to_complex}_complexity'
    cumulative_store_complexity_list = base_dependencies.load_jsonl(f'outputs/dpo/cumulative_stores/{col_to_complex}_complexity.jsonl')
    previous_complexities = pd.DataFrame(cumulative_store_complexity_list)
    complexity_subset = df_w_sentences[~df_w_sentences[col_to_complex].isin(previous_complexities[col_to_complex])].drop_duplicates(subset=[col_to_complex], keep='first').reset_index(drop=True)

    # if any sentences don't have complexity assigned, assign it.
    if len(complexity_subset) > 0:
        print(f'calculating complexities for {len(complexity_subset)} {col_to_complex}')
        complexity_model, complexity_tokenizer = base_dependencies.launch_complexity_model()
        complexity_subset[complexity_col] = base_dependencies.complexity_sentence_list(complexity_model, complexity_tokenizer, list(complexity_subset[col_to_complex]))
        new_cum_store = pd.concat([previous_complexities, complexity_subset[[col_to_complex, complexity_col]]], ignore_index=True).reset_index(drop=True)
        pd.DataFrame(new_cum_store).to_json(f'outputs/dpo/cumulative_stores/{col_to_complex}_complexity.jsonl', orient='records', lines=True)
    else: 
        pass

    previous_complexities = pd.DataFrame(base_dependencies.load_jsonl(f'outputs/dpo/cumulative_stores/{col_to_complex}_complexity.jsonl'))
    df_w_sentences = pd.merge(df_w_sentences, previous_complexities, how='left', on=col_to_complex, validate='m:1')

    return df_w_sentences

def embeddings_sentences(df_w_sentences, col_to_embed):

    embedding_col = f'{col_to_embed}_embedding'
    cumulative_store_embedding_list = base_dependencies.load_jsonl(f'outputs/dpo/cumulative_stores/{col_to_embed}_embedding.jsonl')
    previous_embeddings = pd.DataFrame(cumulative_store_embedding_list)
    embedding_subset = df_w_sentences[~df_w_sentences[col_to_embed].isin(previous_embeddings[col_to_embed])].drop_duplicates(subset=[col_to_embed], keep='first').reset_index(drop=True)

    # if any sentences don't have embedding assigned, assign it.
    if len(embedding_subset) > 0:
        print(f'calculating embedding for {len(embedding_subset)} {col_to_embed}')
        embedding_model = base_dependencies.launch_embedding_model()
        embedding_subset[embedding_col] = base_dependencies.embeddings(list(embedding_subset[col_to_embed]), embedding_model)
        new_cum_store = pd.concat([previous_embeddings, embedding_subset[[col_to_embed, embedding_col]]], ignore_index=True).reset_index(drop=True)
        pd.DataFrame(new_cum_store).to_json(f'outputs/dpo/cumulative_stores/{col_to_embed}_embedding.jsonl', orient='records', lines=True)
    else: 
        pass

    previous_embedding = pd.DataFrame(base_dependencies.load_jsonl(f'outputs/dpo/cumulative_stores/{col_to_embed}_embedding.jsonl'))
    df_w_sentences = pd.merge(df_w_sentences, previous_embedding, how='left', on=col_to_embed, validate='m:1')

    return df_w_sentences

################################################################################ GENERATION HANDLING

# NO LONGER IN USE AFTER IMPROVED SFT
def meta_commentary_filter(simplification):

        smp = simplification.lower()
        if 'vereinfachte version des satzes' in smp:
            return 'vereinfachte version des satzes'
        if 'oder noch einfacher' in smp:
            return 'oder noch einfacher'
        if 'alternativer satz:' in smp:
            return 'alternativer satz:'
        if 'alternativ:' in smp:
            return 'alternativ:'
        if 'vereinfachter satz' in smp:
            return 'vereinfachter satz'
        if 'einfacherer satz' in smp:
            return 'einfacherer satz'
        if 'einfacher satz' in smp:
            return 'einfacher satz'
        if 'ein beispiel für eine erklärung' in smp:
            return 'ein beispiel für eine erklärung'
        if 'oder in einem anderen ausdruck' in smp:
            return 'oder in einem anderen ausdruck'
        if '\nleicht:' in smp:
            return '\nleicht:'
        if '. leicht:' in smp:
            return '. leicht:'
        if '\noder:' in smp:
            return '\noder:'
        if '\n\n' in smp: # used to report alternative simplifications or explain itself with commentary
            return '\n\n'
        if '. - ' in smp: # used to report alternative simplifications
            return '. - '
        if 'kein wandel' in smp:
            return 'kein wandel'
        if '(keine vereinfachung' in smp:
            return '(keine vereinfachung'
        if 'einfacher sprache' in smp:
            return 'einfacher sprache'
        if 'einfachen sprache' in smp:
            return 'einfachen sprache'
        if 'bereits sehr einfach' in smp:
            return 'bereits sehr einfach'
        if 'bereits relativ einfach' in smp:
            return 'bereits relativ einfach'
        if 'bereits einfach' in smp:
            return 'bereits einfach'
        if 'vereinfachten version' in smp:
            return 'vereinfachten version'
        if 'folgendermaßen umgeschrieben' in smp:
            return 'folgendermaßen umgeschrieben'
        if 'der satz' in smp:
            return 'der satz'
        return ''

def eops_count(simp1, simp2):

    # count word-level edit operations (excludes reordering of words)

    s_analysis = list(difflib.Differ().compare(simp1.split(), simp2.split()))
    s_analysis_agnostic = [w[1:].translate(str.maketrans('', '', string.punctuation)).strip().lower() for w in s_analysis if '+' == w[0] or '-' == w[0]]
    s_analysis_agnostic_unique_only =  [w for w in s_analysis_agnostic if s_analysis_agnostic.count(w) == 1]

    return len(s_analysis_agnostic_unique_only)

def left_or_right_in_pair(pair):
    if random.random() > 0.5:
        return (pair[0], pair[1])
    else:
        return (pair[1], pair[0])

def previous_manual_feedback_position_agnostic():
    previous_manual_feedback = pd.DataFrame(base_dependencies.load_jsonl(f'outputs/dpo/cumulative_stores/manual_feedback.jsonl'))
    previous_manual_feedback = previous_manual_feedback.drop_duplicates(keep='last')
    previous_manual_feedback = previous_manual_feedback[['original', 'simp1', 'simp2', 'manual_feedback']]
    previous_manual_feedback_copy = previous_manual_feedback.copy()
    previous_manual_feedback_copy['simp1'] = previous_manual_feedback['simp2']
    previous_manual_feedback_copy['simp2'] = previous_manual_feedback['simp1']
    previous_manual_feedback = pd.concat([previous_manual_feedback, previous_manual_feedback_copy]).reset_index(drop=True)
    return previous_manual_feedback

################################################################################ MANUAL PAIR CHECKING

team_annotator_mapping = {
    '600':'pc02',
    '1200':'pc14',
    '1800':'pc03',
    '2400':'pc04',
    '3000':'pc05',
    '3300':'pc06',
    '3600':'pc07',
    '4200':'pc08',
    '4800':'pc09',
    '5100':'pc10',
    '5400':'pc11',
    '6000':'pc02',
    '6600':'pc12',
    '8000':'pc13',
}

def load_pair_creation_files(latest_team_pairs):
    directory = f"outputs/dpo/pair_creation/{latest_team_pairs}/"
    winning_files = [f for f in os.listdir(directory) if f.startswith("winning_pairs") and f.endswith(".jsonl")]
    losing_files = [f for f in os.listdir(directory) if f.startswith("losing_originals") and f.endswith(".jsonl")]
    dfs1 = []
    dfs2 = []
    for file in winning_files:
        df = pd.read_json(os.path.join(directory, file), lines=True)
        df["creator"] = team_annotator_mapping[file.split('pairs_')[1].split('.')[0]]
        dfs1.append(df)
    for file in losing_files:
        df = pd.read_json(os.path.join(directory, file), lines=True)
        df["creator"] = team_annotator_mapping[file.split('originals_')[1].split('.')[0]]
        dfs2.append(df)
    winning_pairs = pd.concat(dfs1, ignore_index=True) if dfs1 else pd.DataFrame()
    losing_originals = pd.concat(dfs2, ignore_index=True) if dfs2 else pd.DataFrame()
    winning_pairs = winning_pairs[winning_pairs['original'] != 'tmp']
    losing_originals = losing_originals[losing_originals['original'] != 'tmp']

    # somehow this duplicated original sentence made it in
    winning_pairs = winning_pairs[~((winning_pairs['original'] == 'Im Jahr 2019 zählte der Tiergarten Schönbrunn über 2,3 Millionen Besucher, davon viele aus dem Ausland.') & (winning_pairs['creator'] != 'pc09'))]
    return winning_pairs, losing_originals

def access_all_pairs(filedate):

    winning_pairs, losing_originals = load_pair_creation_files(f'team_pairs_{filedate}')
    unique_simplification_pairs = pd.DataFrame(base_dependencies.load_jsonl('outputs/dpo/possible_pairs_20250209/possible_pairs.jsonl'))
    winning_pairs = pd.merge(winning_pairs, unique_simplification_pairs, on=['original', 'simp1', 'simp2'], how='left').reset_index(drop=True)
    winning_pairs = winning_pairs.sort_values(by=['original', 'creator']).drop_duplicates(subset='original', keep='last')
    winning_pairs['simp_pair_id'] = 'r2_' + winning_pairs['simp_pair_id']
    winning_pairs = winning_pairs[['model', 'orig_id', 'original', 'prompt', 'simp_pair_id', 'simp1', 'simp2', 'perp1', 'perp2', 'creator', 'inf_params1', 'inf_params2', 'info_equality']]
    winning_pairs['original_source'] = winning_pairs['orig_id'].apply(lambda x: x.split('_')[1])
    
    kaede_pairs = pd.DataFrame(base_dependencies.load_jsonl(f'outputs/dpo/pair_creation/kaede_pairs_20250128.jsonl'))
    kaede_pairs = kaede_pairs[['model', 'orig_id', 'original', 'prompt', 'simp_pair_id', 'simp1', 'simp2', 'perp1', 'perp2']]
    kaede_pairs['creator'] = 'pc01'
    kaede_pairs['inf_params1'] = 'p0.9t1'
    kaede_pairs['inf_params2'] = 'p0.9t1'
    kaede_pairs['simp_pair_id'] = 'r1_' + kaede_pairs['simp_pair_id']

    all_winning_pairs = pd.concat([winning_pairs, kaede_pairs], axis=0)
    all_winning_pairs = all_winning_pairs.sort_values(by=['orig_id', 'inf_params1'])
    all_winning_pairs = all_winning_pairs.drop_duplicates(subset=['prompt', 'original', 'simp1', 'simp2'], keep='last')
    all_winning_pairs.to_json('outputs/dpo/pair_creation/all_created_pairs.jsonl', orient='records', lines=True)

    return winning_pairs, losing_originals, kaede_pairs, all_winning_pairs

################################################################################ MANUAL PAIR CHECKING

def originals_with_successful_pairs():
    previous_manual_feedback = pd.DataFrame(base_dependencies.load_jsonl(f'outputs/dpo/cumulative_stores/manual_feedback.jsonl'))
    found_originals = set(previous_manual_feedback[previous_manual_feedback['manual_feedback'] == 'w']['original'])
    return found_originals

def add_to_manual_feedback(previous_manual_feedback, orig, s1, s2, feedback):
    simp_pair_dct = {'original':orig, 'simp1':s1, 'simp2':s2, 'manual_feedback':feedback}
    previous_manual_feedback = pd.concat([previous_manual_feedback, pd.DataFrame([simp_pair_dct])], ignore_index=True)
    previous_manual_feedback.to_json('outputs/dpo/cumulative_stores/manual_feedback.jsonl', orient='records', lines=True)
    with open('outputs/dpo/cumulative_stores/manual_feedback.jsonl', 'r') as file:
        file.flush()
        file.close()

################################################################################ PAIR UPLOAD

def add_intra_to_ta(anno_dataframes, annos_clean, num_annos_so_far, already_intrad, already_interd, all_pairs):
    annos_clean = annos_clean[annos_clean['round'] == 'r2']
    annos_clean = pd.merge(annos_clean, already_intrad, how='left', on=['userID', 'simp_pair_id'], indicator=True)
    annos_clean = annos_clean[annos_clean['_merge'] == 'left_only']
    annos_clean = annos_clean[~annos_clean['simp_pair_id'].isin(already_interd)]
    annos_clean = annos_clean.groupby('userID').apply(lambda x: x.sample(n=30)).reset_index(drop=True)[['userID', 'simp_pair_id']]
    for usr in num_annos_so_far:
        intra_for_7pct = int(num_annos_so_far[usr]*0.07)
        intra_for_7pct = intra_for_7pct - len(already_intrad[already_intrad['userID'] == usr])
        if intra_for_7pct < 10:
            intra_for_7pct = 10
            intra_for_7pct = intra_for_7pct - len(already_intrad[already_intrad['userID'] == usr])
        mask = (annos_clean['userID'] != usr)
        intra_sample = annos_clean[annos_clean['userID'] == usr].head(intra_for_7pct).index
        annos_clean = annos_clean[mask | annos_clean.index.isin(intra_sample)]
    annos_clean = pd.merge(all_pairs, annos_clean, how='inner', on=['simp_pair_id'])
    annos_clean = annos_clean[['userID', 'prompt', 'original', 'simp1', 'simp2', 'orig_id', 'simp_pair_id', 'model', 'perp1', 'perp2', 'creator', 'inf_params1', 'inf_params2']]
    annos_clean['origin'] = 'intra'
    for annotator in anno_dataframes.keys():
        if annotator in annos_clean['userID'].unique():
            anno_dataframes[annotator] = pd.concat([annos_clean[annos_clean['userID'] == annotator].drop(columns=['userID']), anno_dataframes[annotator]])
    return anno_dataframes

def get_app_ready(post_dict_sampler_df):
    app_ready = post_dict_sampler_df.copy(deep=True)
    app_ready = app_ready.rename(columns={'simp1':'simplification1','simp2':'simplification2','perp1':'question', 'perp2':'options'})
    try:
        app_ready['answer'] = app_ready['model']
    except:
        pass
    app_ready['question'] = app_ready['question'].astype(str)
    app_ready['options'] = app_ready['options'].astype(str)
    app_ready = app_ready[['original', 'simplification1', 'simplification2', 'question', 'options', 'answer', 'orig_id', 'simp_pair_id']]
    
    return app_ready

################################################################################ PAIR EXPLORATORY ANALYSIS

def dist_len_perp_sentences(pairs_df):

    pairs_df = pairs_df[pairs_df['useable'] != False]
    perplexity_differential = pairs_df['perplexity1'] - pairs_df['perplexity2']

    pairs_df = pairs_df[['simplification1', 'simplification2']]
    pairs_df['s1_split_orig'] = pairs_df['simplification1'].str.split()
    pairs_df['s2_split_orig'] = pairs_df['simplification2'].str.split()
    pairs_df['s1_tokens'] = pairs_df.apply(lambda row: len(row['s1_split_orig']), axis=1)
    pairs_df['s2_tokens'] = pairs_df.apply(lambda row: len(row['s2_split_orig']), axis=1)
    tokens = pd.concat([pairs_df['s1_tokens'], pairs_df['s2_tokens']])

    pairs_df['tokens_diff'] = pairs_df['s1_tokens'] - pairs_df['s2_tokens']
    pairs_df['tokens_diff']

    plt.figure(figsize=(15,5))
    plt.subplot(1, 3, 1)
    tokens.plot(kind='hist', density=False, alpha=0.7)
    plt.title('Simplification Length')
    plt.ylabel(' ')

    plt.subplot(1, 3, 2)
    pairs_df['tokens_diff'].plot(kind='hist', density=False, alpha=0.7)
    plt.title('Simplification Length Differential')
    plt.ylabel(' ')
    
    plt.subplot(1, 3, 3)
    perplexity_differential.plot(kind='hist', density=False, alpha=0.7)
    plt.title('Perplexity Differential')
    plt.ylabel(' ')
    
    plt.show()

################################################################################ ANNOTATION HANDLING

def pilot_to_standard_id_dict():
    pid_to_id_df = pd.read_csv('outputs/dpo/original_sentence_uid.csv')

    pid_to_id_df = pid_to_id_df.dropna(subset=['pilot_id'])
    pid_to_id = pid_to_id_df.set_index('pilot_id')['id'].to_dict()

    pid_to_id = {k: (v if pd.notna(v) else k) for k, v in pid_to_id.items()}

    return pid_to_id

# identify problematic pairs
## typos or errors from a user session
user_issues = []
user_issues.append(('Wenn man einen Brand-Anschlag plant und durchführen lässt, kann man sogar die Todesstrafe bekommen.', 'Für einen Brand-Anschlag kann man sogar die Todesstrafe bekommen. Das ist ein sehr strenges Straf-Regel.'))
user_issues.append(('Polizisten machen Fehler, genauso wie Golf-Spieler. Wegen Stress oder Ermüdung.', 'Polizisten machen Fehler, genauso wie Spieler in dem Spielen Golf. Der Druck macht sie nervös. Und Nervosität führt zu Fehlern.'))
user_issues.append(('In Österreich gibt es eine Regierungs-Krise. Die Ursache ist das sogenannte Ibiza-Video.', 'In Österreich gibt es eine Regierungs-Krise wegen dem Ibiza-Video. Das Video zeigt einen Politiker bei einer illegalem Geschäft.'))
user_issues.append(('Man darf wieder überall hingehen. Das gilt auch nicht nur für Einkaufen oder Arbeit.', 'Nun kann man überall hin. Man darf zum Spielen und zum Feiern in die Natur oder zum Bade gehen.'))
user_issues.append(('Tausende Menschen ehren Niki Lauda.', 'Tausende Menschen haben Niki Lauda ehrt. Sie wollten zeigen, dass sie Niki Lauda schätzten.'))
user_issues.append(('Schon 3 Mal wurde der Star Douglas von der Oscar-Academy nominiert. Der Begriff „nominiert“ heißt in Deutsch: Ausgezeichnet.', 'Douglas konnte den Oscar bisher 3 Mal gewinnen. Er war 3 Mal nominiert.'))
user_issues.append(('Von und nach Italien gab es am Mittwoch auch keinen Zug-Verkehr mehr.', 'Am Mittwoch hat die Bahn von und nach Italien nicht mehr gefahren.'))

## typos or errors from expert annotator document
# one- or twosided grammar error
expert_issues = []
expert_issues.append(('Baby-Elefant nennt man den Mindest-Abstand. Er gibt es seit der Corona-Krise. Bei Baby-Elefant soll man die Entfernung zu anderen Menschen haben. Man soll 1,5 Meter von ihnen fern bleiben.', 'Der Begriff "Baby-Elefant" soll den Mindest-Abstand bezeichnen, den man seit der Corona-Krise in der Öffentlichkeit einhalten muss.'))
expert_issues.append(('In vielen Orten haben die Flüsse über die Ufer getreten. Diese überfluteten ganze Wohn-Gebiete und Straßen.', 'Viele Flüsse traten über die Ufer und überschwemmten Städte und Wohn-Gebiete.'))
expert_issues.append(('Die Polizei denkt, dass es ein Terror-Anschlag war.', 'Die Polizei glaubt, dass es einen Terror-Anschlag war.'))
expert_issues.append(('Der Fix-Kosten-Zuschuss ist eine Hilfe für die Geschäfte. Der Schutz-Schirm soll zusätzlich helfen, sagte Blümel. Der Schutz-Schirm bietet im Vorhinein Planung-Sicherheit.', 'Es solle sozusagen eine Hilfe im Nachhinein sein. Ein anderer Schutzschirm soll helfen, im Voraus zu wissen, wie viel Geld man erhält. Damit will Blümel die Menschen beschützen.'))
expert_issues.append(('In Deutschland vermisste ein Mann sein Auto. Er suchte das Auto 3 Wochen lang.', 'Ein Mann in Deutschland verlor seinen Auto. Der Mann suchte nach seinem Auto. Er fand es erst 3 Wochen später wieder.'))
expert_issues.append(('LASK gewinnt überraschend in der Champions League.', 'Das Team LASK wird an der Champions League spielen. Das war nicht vorher abzusehen.'))
expert_issues.append(('Eisbären sind bei der Geburt so groß wie ein kleines Meerschweinchen und wiegen ungefähr so viel wie ein Meerschweinchen.', 'Junge Eisbären kommen etwa ein halbes Kilo auf die Welt und sind dann so groß wie ein Meerschweinchen.'))
expert_issues.append(('In Österreich hat sich in den letzten 24 Stunden 272 Menschen mit dem Corona-Virus angesteckt.', 'In den letzten 24 Stunden gab es in Österreich 272 Neu-Infektionen mit dem Corona-Virus.'))
expert_issues.append(('Der Burgenländische Landtag hat an Montag gewählt. Er wählte Hans Peter Doskozil erneut zum Landeshauptmann. Doskozil gehört der SPÖ an.', 'Der Burgenländische Landtag hat Hans Peter Doskozil wiedergewählt. Hans Peter Doskozil ist die Landeshauptmann von Burgenland. Er stammt von der Partei SPÖ.'))
expert_issues.append(('Diese Partei wird das Team von Frau von der Leyen unterstützen.', 'In dem Licht wird ihre Fraktion Team von Ursula von der Leyen unterstützen.'))
expert_issues.append(('Schon 3 Mal wurde der Star Douglas von der Oscar-Academy nominiert. Der Begriff „nominiert“ heißt in Deutsch: Ausgezeichnet.', 'Douglas konnte den Oscar bisher 3 Mal gewinnen. Er war 3 Mal nominiert.'))
expert_issues.append(('Teilweise fehlte es auch an Bewegung. Foda wirkte unzufrieden.', 'Sie hatten keinen Bewegung. Man sah Foda an, dass er unglücklich war.'))
expert_issues.append(('Viele Frauen in Österreich fühlen sich gestresst. Grund dafür sind die Home-Office-Arbeit und das Home-Schooling für die Kinder. Laut AK ist das die häufigste Antwort.', 'Home-Office und Home-Schooling stressten Frauen am meisten. Das sagt der AK.'))
expert_issues.append(('Das Nein zum Freihandels-Abkommen hat sich auch der Handels-Verband gefreut. Er nannte das Nein für den Handel wichtig.', 'Daher ist auch der Handels-Verband zufrieden. Der Handels-Verband hat das nein zum Abkommen als Sieg gewertet.'))
expert_issues.append(('Tausende Menschen ehren Niki Lauda.', 'Tausende Menschen haben Niki Lauda ehrt. Sie wollten zeigen, dass sie Niki Lauda schätzten.'))
expert_issues.append(('Ausgenommen sind Osttirol und Kinder bis 10 Jahre. Diese brauchen keinen Test.', 'Ausgenommen sind dabei die Gebiete Osttirol und Kindergarten- und Schüler bis zehn Jahre. Diese müssen keinen Test vorweisen.'))
expert_issues.append(('Man muss den Wunsch nach einem Papamonat dem Arbeitgeber vorher mitteilen. Dafür gibt es eine bestimmte Frist.', 'Für einen Papi-Monat gibt es jedoch Vorschriften. Arbeitgeber und Arbeitnehmer müssen ihn mitteilen. Das darf frühestens drei Monate im Voraus.'))
expert_issues.append(('Im Freien darf eine Gruppe von bis zu 12 Menschen zusammen sein.', 'Im Freien können bis zu 12 Personen miteinander treffen.'))
expert_issues.append(('In den Jahren davor ist dieser Anteil immer höher geworden. Im Vorjahr lag er schon bei 56 Prozent.', 'Dieser Anteil ist in den letzten Jahren immer höher geworden. Im Vorjahr war es schon 56 Prozent.'))
expert_issues.append(('Der Flieder blüht schon jetzt in wärmeren Regionen von Österreich. Das ist früher, als normal. Er blüht sogar schon mehr als eine Woche früher.', 'Der Flieder blüht in Österreich jetzt schon sehr früh. In den warmen Regionen ist der Flieder schon seit über einer Woche im Blühen.'))
expert_issues.append(('Die Staatsanwaltschaft machte eine Obduktion. Außerdem soll man herausfinden, ob die Buslenkerin etwas getrunken hat. Das heißt, es soll ein toxikologisches Gutachten geben.', 'Die Staatsanwaltschaft in Graz machte auch eine Obduktion und ein toxisches Gutachten. Das heißt, die Leiche von der Bus-Fahrerin wird untersucht.'))
expert_issues.append(('Fleisch im Supermarkt ist in Österreich am teuersten in der EU.', 'Österreich hat im EU-Vergleich am teuersten Fleisch.'))
expert_issues.append(('Die EU-Kommissions-Chefin von der Leyen rechnet mit positivem Bescheid.', 'Von der Leyen rechnet mit einer positiven Entscheidung.'))

global_issues = set(user_issues + expert_issues)

def identify_problem_pairs(row):
    if (row['simplification1'], row['simplification2']) in global_issues or (row['simplification2'], row['simplification1']) in global_issues:
        row['issue'] = 'y'
    else:
        row['issue'] = 'n'
    return row

def collect_annotations(phase):

    pid_to_id = pilot_to_standard_id_dict()

    if phase == 'anno':
        anno_directory = 'data/annotations/'

    combined_annotations = []
    for filename in os.listdir(anno_directory):
        if filename.endswith('.jsonl'):
            curr_pref_annotations = base_dependencies.load_jsonl(os.path.join(anno_directory, filename))
            if 'A_2024-12-11' in filename or 'A_2024-12-13' in filename:
                a_ids = pd.read_csv('data/ATS pairs/raw_ta01_anno_A.csv')
                i = 0
                for dict in curr_pref_annotations:
                    dict['answer'] = a_ids.iloc[i]['answer']
                    i += 1
            if 'B_2024-12-11' in filename or 'B_2024-12-13' in filename:
                b_ids = pd.read_csv('data/ATS pairs/raw_ta01_anno_B.csv')
                i = 0
                for dict in curr_pref_annotations:
                    dict['answer'] = b_ids.iloc[i]['answer']
                    i += 1
            if 'C_2024-12-13' in filename:
                c_ids = pd.read_csv('data/ATS pairs/raw_ta01_anno_C.csv')
                i = 0
                for dict in curr_pref_annotations:
                    dict['answer'] = b_ids.iloc[i]['answer']
                    i += 1
            if 'D_2024-12-13' in filename:
                b_ids = pd.read_csv('data/ATS pairs/raw_ta01_anno_D.csv')
                i = 0
                for dict in curr_pref_annotations:
                    dict['answer'] = b_ids.iloc[i]['answer']
                    i += 1
            if filename == 'labeled_Ta10_anno_A_2025-02-12-14-40-40.jsonl':
                curr_pref_annotations = curr_pref_annotations[:240]
            if filename == 'labeled_ea02_anno_A_2025-03-12-13-02-20.jsonl':
                curr_pref_annotations = curr_pref_annotations[0:838]
            if filename == 'labeled_ta12_anno_A_2025-03-13-14-22-44.jsonl':
                curr_pref_annotations = curr_pref_annotations[0:450]
            if filename == 'labeled_ea02_anno_A_2025-03-13-10-10-08.jsonl':
                curr_pref_annotations = curr_pref_annotations[48:]
            if filename == 'labeled_ta12_anno_A_2025-04-10-11-08-28.jsonl':
                curr_pref_annotations = curr_pref_annotations[0:162]
            for anno_dct in curr_pref_annotations:
                anno_dct['filename'] = filename
                anno_dct['timestamp'] = datetime.strptime(filename.split('_')[-1].split('.')[0], "%Y-%m-%d-%H-%M-%S")

                if 'question' in anno_dct:
                    anno_dct['perplexity1'] = anno_dct.pop('question')

                if 'options' in anno_dct:
                    anno_dct['perplexity2'] = anno_dct.pop('options')[0]

                if 'id' not in anno_dct:
                    anno_dct['id'] = anno_dct.pop('answer')

                # if annotation is from pilot and pilot sentence is also in post-pilot sample sentences,
                # change id to post-pilot id (helps to later ensure no unintended crossover)
                # anything still with 'pilot' in id after this was for a sentence ONLY in pilot
                if 'pilot' in anno_dct['id'] and anno_dct['id'] in pid_to_id:
                        anno_dct['id'] = pid_to_id[anno_dct['id']]

            combined_annotations += curr_pref_annotations

    combined_annotations = pd.DataFrame(combined_annotations)

    # split simplifications, add date info
    combined_annotations = combined_annotations[['userID', 'original', 'simplifications', 'preference', 'filename', 'id', 'timestamp', 'perplexity1', 'perplexity2']]
    combined_annotations[['simplification1', 'simplification2']] = pd.DataFrame(combined_annotations['simplifications'].tolist(), index=combined_annotations.index)
    combined_annotations['day'] = combined_annotations['timestamp'].dt.date
    combined_annotations['earliest_timestamp_by_day'] = combined_annotations.groupby(['userID', 'day'])['timestamp'].transform('min')
    combined_annotations['latest_timestamp_by_day'] = combined_annotations.groupby(['userID', 'day'])['timestamp'].transform('max')
    
    # attach simplification id
    winning_pairs = pd.DataFrame(base_dependencies.load_jsonl(f'data/ATS pairs/all_created_pairs.jsonl'))
    eval_pairs = pd.read_json('data/ATS pairs/eval_pairs_w_metadata.jsonl', lines=True)
    winning_pairs = winning_pairs[['original', 'simp1', 'simp2', 'simp_pair_id', 'info_equality']]
    eval_pairs = eval_pairs[['original', 'simplification1', 'simplification2', 'simp_pair_id', 'info_equality']]
    eval_pairs = eval_pairs.rename(columns={'simplification1':'simp1', 'simplification2':'simp2'})
    winning_pairs = pd.concat([winning_pairs, eval_pairs])
    winning_pairs['ordering'] = 'original'

    simps_swapped = winning_pairs.copy()
    simps_swapped[['simp1', 'simp2']] = simps_swapped[['simp2', 'simp1']]
    simps_swapped['ordering'] = 'reversed'

    winning_pairs_copy = winning_pairs.copy()
    # problem pair: one of the final evals is actually an OvS pair because sft checkpoints occasionally output original. Dropping it from OvS set so eval metadata can be merged in properly.
    winning_pairs_copy = winning_pairs_copy[winning_pairs_copy['original'] != '"Mit der Impfung ist der Anfang für den Sieg gegen die Pandemie eingeleitet", sagte Bundeskanzler Sebastian Kurz.']

    ovs_left1 = winning_pairs_copy.copy()
    ovs_left1['simp1'] = ovs_left1['original']
    ovs_left1['ordering'] = 'original'
    ovs_left1['simp_pair_id'] = ovs_left1['simp_pair_id'].str.replace('pid', 'ovs')

    ovs_left2 = winning_pairs_copy.copy()
    ovs_left2['simp2'] = ovs_left2['simp1']
    ovs_left2['simp1'] = ovs_left2['original']
    ovs_left2['ordering'] = 'original'
    ovs_left2['simp_pair_id'] = ovs_left2['simp_pair_id'].str.replace('pid', 'ovs')

    ovs_right1 = winning_pairs_copy.copy()
    ovs_right1['simp2'] = ovs_right1['original']
    ovs_right1['ordering'] = 'reversed'
    ovs_right1['simp_pair_id'] = ovs_right1['simp_pair_id'].str.replace('pid', 'ovs')

    ovs_right2 = winning_pairs_copy.copy()
    ovs_right2['simp1'] = ovs_right2['simp2']
    ovs_right2['simp2'] = ovs_right2['original']
    ovs_right2['ordering'] = 'reversed'
    ovs_right2['simp_pair_id'] = ovs_right2['simp_pair_id'].str.replace('pid', 'ovs')

    winning_pairs = pd.concat([winning_pairs, simps_swapped, ovs_left1, ovs_left2, ovs_right1, ovs_right2], ignore_index=True)

    # problem pair: somehow ta01 got an old version of pair for 'Die Grenzen in Slowenien werden für EU-Bürger geöffnet.'
    combined_annotations = combined_annotations[~((combined_annotations['original'] == 'Die Grenzen in Slowenien werden für EU-Bürger geöffnet.') & (combined_annotations['userID'] == 'ta01'))]

    # remove rows with missing originals
    combined_annotations = combined_annotations[combined_annotations['original'] != '']

    # merge to bring in simp_pair_id
    combined_annotations = pd.merge(left=combined_annotations, right=winning_pairs, how='left', left_on=['original', 'simplification1', 'simplification2'], right_on=['original', 'simp1', 'simp2'], indicator=True)
    copy_for_return = combined_annotations.copy()

    combined_annotations['simp_pair_id'] = combined_annotations['simp_pair_id'].fillna(combined_annotations['id'])
    assert len(combined_annotations[combined_annotations['simp_pair_id'].isna()]) == 0
    # one day had issue where ovs pairs were blank
    combined_annotations = combined_annotations[combined_annotations['original'] != '']
    # necessary userid edits based on day 1
    date_filter = combined_annotations['day'] == pd.to_datetime('2025-01-29').date()
    if date_filter.any():  
        combined_annotations.loc[date_filter & (combined_annotations['userID'] == 'ta11'), 'userID'] = 'ta09'  
        combined_annotations.loc[date_filter & (combined_annotations['userID'] == 'ta13'), 'userID'] = 'ta01'  
        combined_annotations.loc[date_filter & (combined_annotations['userID'] == 'ta12'), 'userID'] = 'ta10'  
        combined_annotations = combined_annotations[~(date_filter & (combined_annotations['userID'] == 'ta10'))]

    # on day 1, for the list below, ta03 would submit one pair, change his choice, and submit again. He would only change once and keep it the rest of the time.
    # ['pid_apa166797', 'pid_apa167045', 'pid_apa167232', 'pid_apa86106', 'pid_apa166536', 'pid_apa85947', 'pid_apa87001', 'pid_apa7884', 'pid_apa168766', 'pid_apa8608', 'pid_apa87824', 'pid_apa168414', 'pid_apa8301', 'pid_apa86106', 'pid_apa168766', 'pid_apa7884', 'pid_apa8301', 'pid_apa168414', 'pid_apa8608', 'pid_apa166536', 'pid_apa166797', 'pid_apa87824', 'pid_apa167045', 'pid_apa167232', 'pid_apa87001', 'pid_apa85947']

    # remove instances of ovs for ea03 and ea01 on March 10 2025
    #combined_annotations = combined_annotations[~((combined_annotations['day'] == pd.to_datetime('2025-03-10').date()) & (combined_annotations['type'].isin(['ovs_right', 'ovs_left'])) & (combined_annotations['userID'].str.contains('ea')))]

    # deduplicate at the FILE level
    # combined_annotations = combined_annotations.sort_values(by=['userID', 'timestamp', 'original', 'simp_pair_id']).drop_duplicates(subset=['userID', 'original', 'simp_pair_id', 'preference', 'timestamp'], keep='last')
    combined_annotations = combined_annotations[['userID', 'preference', 'filename', 'timestamp', 'perplexity1', 'perplexity2', 'id', 'original', 'simplification1', 'simplification2', 'day', 'earliest_timestamp_by_day', 'latest_timestamp_by_day', 'simp_pair_id', 'ordering', 'info_equality']]
    
    combined_annotations = combined_annotations.rename(columns={'preference':'raw_preference', 'perplexity1':'raw_perplexity1', 'perplexity2':'raw_perplexity2', 'simplification1':'raw_simplification1', 'simplification2':'raw_simplification2'})

    print(len(combined_annotations))

    # standardize ordering of inter, intra, ovs pairs
    reversed_pairs = combined_annotations["ordering"] == "reversed"

    combined_annotations.loc[reversed_pairs, ["perplexity1", "perplexity2"]] = combined_annotations.loc[reversed_pairs, ["raw_perplexity2", "raw_perplexity1"]].values
    combined_annotations.loc[~reversed_pairs, ["perplexity1", "perplexity2"]] = combined_annotations.loc[~reversed_pairs, ["raw_perplexity1", "raw_perplexity2"]].values

    combined_annotations.loc[reversed_pairs, ["simplification1", "simplification2"]] = combined_annotations.loc[reversed_pairs, ["raw_simplification2", "raw_simplification1"]].values
    combined_annotations.loc[~reversed_pairs, ["simplification1", "simplification2"]] = combined_annotations.loc[~reversed_pairs, ["raw_simplification1", "raw_simplification2"]].values

    combined_annotations["preference"] = combined_annotations["raw_preference"]

    mask_valid_pref = reversed_pairs & combined_annotations["raw_preference"].notna()
    combined_annotations.loc[mask_valid_pref, "preference"] = (combined_annotations.loc[mask_valid_pref, "raw_preference"].str.replace("1", "X").str.replace("2", "1").str.replace("X", "2"))

    combined_annotations["userID"] = combined_annotations["userID"].str.lower()

    #combined_annotations["simp_pair_id"] = np.where(combined_annotations["simp_pair_id"].str.contains('r3_ovs'), combined_annotations["simp_pair_id"].str.replace('r3_ovs', 'r3_pid'), combined_annotations["simp_pair_id"])
    
    combined_annotations = combined_annotations.apply(identify_problem_pairs, axis=1)

    return combined_annotations


################################################################################ EVALUATION

def length_and_perplexity(anno_df):
    anno = anno_df[['simplifications', 'preference', 'perplexity1', 'perplexity2', 'id']]
    anno['s1_len'] = anno.apply(lambda row: len(row['simplifications'][0].split()), axis=1)
    anno['s2_len'] = anno.apply(lambda row: len(row['simplifications'][1].split()), axis=1)
    print(anno.iloc[0])
    anno['wc_diff'] = anno.apply(lambda row: row['s1_len'] - row['s2_len'] if row['preference'] == 'Vereinfachung 1' else row['s2_len'] - row['s1_len'], axis=1)
    anno['p_diff'] = anno.apply(lambda row: float(row['perplexity1']) - float(row['perplexity2']) if row['preference'] == 'Vereinfachung 1' else float(row['perplexity2']) - float(row['perplexity1']), axis=1)
    return anno.sort_values('wc_diff')

def lap_violins(target, expert, column):
    target['source'] = 'Target User'
    expert['source'] = 'Expert User'
    anno = pd.concat([target, expert], ignore_index=True)
    sns.violinplot(x=anno['source'], y=anno[column], order=['Target User', 'Expert User'], orient='v')

    if column == 'wc_diff':
        ylabel_str = 'Word Count of Preferred - \n Word Count of Non-Preferred \n\n (<-  Preferred is Shorter            Preferred is Longer ->)'
        title_str = 'Preferred vs. Non-Preferred Length Differential by Annotator'
    if column == 'p_diff':
        ylabel_str = 'Perplexity of Preferred - \n Perplexity of Non-Preferred \n\n (<- Preferred is Lower Perplexity    Preferred is Higher Perplexity ->)'
        title_str = 'Preferred vs Non-Preferred Perplexity Differential by Annotator'
    plt.ylabel(ylabel_str)
    plt.title(title_str)
    plt.xlabel('Annotation Source')
    plt.show()

def inter_annotator_agreement(df, detail=True, ret = False):

    pairs_by_user = df['userID'].value_counts()  

    df = df[~df['preference'].isna()]
    df = df.sort_values(by=['userID'])
    annotations_by_user = df['userID'].value_counts()  
    overlapping_counts = {}  
    iaa_scores = {}  

    for user1, user2 in combinations(df['userID'].unique(), 2):

        df1 = df[df['userID'] == user1].set_index(['original', 'simp_pair_id'])[['preference']]  
        df2 = df[df['userID'] == user2].set_index(['original', 'simp_pair_id'])[['preference']]  

        common = df1.join(df2, lsuffix='_1', rsuffix='_2', how='inner').dropna()  

        if not common.empty:  
            overlapping_counts[(user1, user2)] = len(common)  
            iaa_scores[(user1, user2)] = cohen_kappa_score(common['preference_1'], common['preference_2'])

    anno_frac_by_user = [f"{entry}: {annotations_by_user.get(entry, 0)} / {pairs_by_user.get(entry, 0)}" for entry in pairs_by_user.index]
    if detail:
        print("Pairs annotated by user:")
        for line in anno_frac_by_user:
            print(line)
        print(sum(annotations_by_user), "total")
        print("\nIAA (Cohen's Kappa, pairwise):")  
    for key in iaa_scores:
        print(f"{key}: {iaa_scores[key]:.3f} ({overlapping_counts[key]} obs)")
    if ret == True:
        return (iaa_scores[key], overlapping_counts[key])

################################################################################ DPO IMPLEMENTATION

save_from_dpo = [['sft_disco_llama8b_checkpoint2800', 135, 'ea', 'all'],
       ['sft_disco_llama8b_checkpoint2800', 270, 'ta', 'all'],
       ['sft_llama8b_checkpoint2400', 165, 'ea', 'all'],
       ['sft_llama8b_checkpoint2400', 180, 'ta', 'all'],
       ['sft_leolm_mistral7b_checkpoint1600', 285, 'ea', 'all'],
       ['sft_leolm_mistral7b_checkpoint1600', 195, 'ta', 'all'],
       ['sft_disco_llama8b_checkpoint2800', 210, 'ea', 'equality'],
       ['sft_disco_llama8b_checkpoint2800', 150, 'ta', 'equality'],
       ['sft_llama8b_checkpoint2400', 15, 'ea', 'equality'],
       ['sft_llama8b_checkpoint2400', 45, 'ta', 'equality'],
       ['sft_leolm_mistral7b_checkpoint1600', 210, 'ea', 'equality'],
       ['sft_leolm_mistral7b_checkpoint1600', 105, 'ta', 'equality'],
       ['sft_disco_llama8b_checkpoint2800', 90, 'ea', 'model'],
       ['sft_disco_llama8b_checkpoint2800', 15, 'ta', 'model'],
       ['sft_llama8b_checkpoint2400', 60, 'ea', 'model'],
       ['sft_llama8b_checkpoint2400', 75, 'ta', 'model'],
       ['sft_leolm_mistral7b_checkpoint1600', 60, 'ea', 'model'],
       ['sft_leolm_mistral7b_checkpoint1600', 30, 'ta', 'model'],
       ['sft_disco_llama8b_checkpoint2800', 150, 'ea', 'interAA'],
       ['sft_disco_llama8b_checkpoint2800', 60, 'ta', 'interAA'],
       ['sft_llama8b_checkpoint2400', 45, 'ea', 'interAA'],
       ['sft_llama8b_checkpoint2400', 120, 'ta', 'interAA'],
       ['sft_leolm_mistral7b_checkpoint1600', 90, 'ea', 'interAA'],
       ['sft_leolm_mistral7b_checkpoint1600', 105, 'ta', 'interAA'],
       ['sft_disco_llama8b_checkpoint2800', 105, 'ea', 'intraAA'],
       ['sft_disco_llama8b_checkpoint2800', 75, 'ta', 'intraAA'],
       ['sft_llama8b_checkpoint2400', 75, 'ea', 'intraAA'],
       ['sft_llama8b_checkpoint2400', 15, 'ta', 'intraAA'],
       ['sft_leolm_mistral7b_checkpoint1600', 105, 'ea', 'intraAA'],
       ['sft_leolm_mistral7b_checkpoint1600', 75, 'ta', 'intraAA'],
       ['sft_disco_llama8b_checkpoint2800', 15, 'ea', 'groupX'],
       ['sft_llama8b_checkpoint2400', 255, 'ea', 'groupX'],
       ['sft_leolm_mistral7b_checkpoint1600', 135, 'ea', 'groupX']]

def access_maximal_dev_step_dpo(model, group, variant):
    for factor in save_from_dpo:
        if factor[0] == model and factor[2] == group and factor[3] == variant:
            return factor[1]
        
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_train_dpotrainer(model, tokenizer, train_dataset, eval_dataset, holdout_dataset, max_seq_length, model_name, variant, user_set, paramix, train_or_test, save_strategy):

    set_seed(5)

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

    per_device_train_batch_size = int(bs / 2)
    gradient_accumulation_steps = ga_steps
    
    #if variant == 'all':
    #    eval_step_size = 13 # every 104 pairs
    #elif len(train_dataset) < 1500:
    #    eval_step_size = int(0.095 * len(train_dataset) / per_device_train_batch_size)
    #else:
    #    eval_step_size = int(0.0498 * len(train_dataset) / per_device_train_batch_size)
    eval_step_size = 15 # every 120 pairs

    print(f"Calculated eval_step_size: {eval_step_size}")

    training_args = DPOConfig(
        output_dir=tmp_dir,
        logging_dir=tmp_dir,
        num_train_epochs=1,
        gradient_accumulation_steps = gradient_accumulation_steps,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=8,
        warmup_ratio=0.1,
        learning_rate=float(lr)*gpu_count,
        fp16=True,
        optim="paged_adamw_32bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        report_to="wandb",
        max_grad_norm=1,
        logging_strategy="no",
        eval_strategy="steps",
        eval_steps = eval_step_size,
        max_length=max_seq_length,
        run_name=f"{model_name}_dpo",
        seed = 5,
        save_strategy=save_strategy,
        save_steps = eval_step_size,
        model_adapter_name = 'dpo_adaptor',
        do_predict = True,
        load_best_model_at_end = True, 
        metric_for_best_model = 'eval_rewards/accuracies',
        greater_is_better = True,
    )

    wandb.init(project="mnlp", name="dpo_" + model_name)

    if train_or_test == 'train':
        trainer = DPOTrainer(
            model = model, 
            tokenizer = tokenizer,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            args = training_args,
            peft_config=peft_config
        )
    if train_or_test == 'test':
        trainer = DPOTrainer(
            model = model, 
            tokenizer = tokenizer,
            train_dataset = train_dataset,
            eval_dataset = holdout_dataset,
            args = training_args,
            peft_config=peft_config
        )
        
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print("\nInitialized DPO Trainer...")
    try:
        print(f'pad token: {tokenizer.pad_token}')
    except:
        print('no pad token')
    
    try:
        print(f'pad side: {tokenizer.padding_side}')
    except:
        print('no padding side')

    print("\nStart training...")

    torch.cuda.empty_cache()
    
    base_dependencies.print_trainable_parameters(model)

    model.train()

    trainer.train()

    base_dependencies.print_trainable_parameters(model)
    
    print("\nDPO completed.")

    print("Trainer best model checkpoint:", trainer.state.best_model_checkpoint)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    accelerator.wait_for_everyone()

    model = trainer.model
    model.eval()

    if accelerator.is_local_main_process:

        base_dependencies.print_trainable_parameters(model)
        print(tmp_dir + model_name)
        trainer.save_model(tmp_dir + model_name)
        if train_or_test == 'train':
            tokenizer.save_pretrained(tmp_dir + model_name)
        trainer.state.save_to_json(tmp_dir + model_name + '/trainer_state.json')
        del model
        del trainer
        del tokenizer
        #del state_dict

    total_train_steps = (len(train_dataset)) // (per_device_train_batch_size * gradient_accumulation_steps) + 1
    print(total_train_steps)

    for model_version in os.listdir(tmp_dir):
        print(model_version, model_name)
        if model_version == model_name:
            if train_or_test == 'train':
                shutil.copy(f"{tmp_dir}/{model_version}/trainer_state.json",f"outputs/dpo_dev_eval/trainer_state_{model_name}_data{variant}_{user_set}.json")
                adaptor_save_direc = f"outputs/models/{paramix}/dpo_all_adaptors/{model_name}_data{variant}_{user_set}"
                shutil.copytree(os.path.join(tmp_dir, model_version), adaptor_save_direc, dirs_exist_ok=True)
            if train_or_test == 'test':
                shutil.copy(f"{tmp_dir}/{model_version}/trainer_state.json",f"outputs/dpo_test_eval/trainer_state_{model_name}_data{variant}_{user_set}.json")

        if train_or_test == 'train':
            if 'checkpoint-' in model_version:
                if [model_name, int(model_version.replace('checkpoint-', '')), user_set, variant] in save_from_dpo:
                    print('saving win rate maximizer')
                    print(model_version)
                    model = AutoPeftModelForCausalLM.from_pretrained(
                                    tmp_dir + model_version,
                                    torch_dtype=torch.float16,
                                    low_cpu_mem_usage=True,
                    )
                    tokenizer = AutoTokenizer.from_pretrained(tmp_dir + model_version)
                    #print('pre merge and unload:')
                    #base_dependencies.print_trainable_parameters(model)
                    merged_model = model.merge_and_unload()
                    #print('post merge and unload:')
                    #base_dependencies.print_trainable_parameters(merged_model)
                    #print(model_version, model_name, gpu_count)
                    merged_model.save_pretrained(f"outputs/models/{paramix}/dpo_{model_name}_data{variant}_{user_set}/{model_version}")
                    tokenizer.save_pretrained(f"outputs/models/{paramix}/dpo_{model_name}_data{variant}_{user_set}/{model_version}")

    accelerator.wait_for_everyone()

def run_init_dpotrainer(model, tokenizer, train_dataset, eval_dataset, max_seq_length, model_name, variant, user_set, paramix, save_strategy):

    set_seed(5)

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

    per_device_train_batch_size = int(bs/2)
    gradient_accumulation_steps = ga_steps
    
    #if variant == 'all':
    #    eval_step_size = 13 # every 104 pairs
    #elif len(train_dataset) < 1500:
    #    eval_step_size = int(0.095 * len(train_dataset) / per_device_train_batch_size)
    #else:
    #    eval_step_size = int(0.0498 * len(train_dataset) / per_device_train_batch_size)
    eval_step_size = 1

    training_args = DPOConfig(
        output_dir=tmp_dir,
        logging_dir=tmp_dir,
        num_train_epochs=1,
        gradient_accumulation_steps = gradient_accumulation_steps,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=8,
        warmup_ratio=0.1,
        learning_rate=float(lr)*gpu_count,
        fp16=True,
        optim="paged_adamw_32bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        report_to="wandb",
        max_grad_norm=1,
        logging_strategy="no",
        eval_strategy="steps",
        eval_steps = eval_step_size,
        max_length=max_seq_length,
        run_name=f"{model_name}_dpo",
        seed = 5,
        save_strategy='no',
        save_steps = eval_step_size
    )

    trainer = DPOTrainer(
        model = model, 
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        args = training_args,
        peft_config=peft_config
    )

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    model.train()
    trainer.train()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:

        base_dependencies.print_trainable_parameters(model)
        print(tmp_dir + model_name)
        trainer.save_model(tmp_dir + model_name)
        tokenizer.save_pretrained(tmp_dir + model_name)
        trainer.state.save_to_json(tmp_dir + model_name + '/trainer_state_init.json')
        del model
        del trainer
        del tokenizer
        #del state_dict

    for model_version in os.listdir(tmp_dir):
        if model_version == model_name:
            shutil.copy(f"{tmp_dir}/{model_version}/trainer_state_init.json",f"outputs/dpo_dev_eval/trainer_state_init_{model_name}_data{variant}_{user_set}.json")


################################################################################ OUTDATED

# subset pairs for pilot

# dpo_original_sentences = base_dependencies.load_jsonl('outputs/dpo/for_pair_generation.jsonl')
# def sample(seed):
#    random.seed(seed)
#    dpo_original_sentences = base_dependencies.load_jsonl('outputs/dpo/for_pair_generation.jsonl')
#    dpo_pilot_sentences = random.sample(dpo_original_sentences, 105)
#    return dpo_pilot_sentences

#global_min = 100
#best_seed = 5
#for i in range(0, 500):
#    dpo_pilot_sentences = sample(i)
#    wd = wasserstein_distance(wcs([dict['original'] for dict in dpo_pilot_sentences]),  wcs([dict['original'] for dict in dpo_original_sentences]))
#    if wd < global_min:
#        best_seed = i
#        global_min = wd
#        print(f'wasserstein distance for seed {i}:', wd)

#dpo_pilot_sentences = sample(best_seed)
#plt_histogram(wcs([dict['original'] for dict in dpo_pilot_sentences]))

# pair generation from openai - dpo_pilot_sentences is list of dicts (json read)

#openai_api = False
#if openai_api == True:
#    dpo_dependencies.pair_generation_sampling_openai(dpo_pilot_sentences, 'pilot_generations_gpt4o_v4.jsonl')
#else:
#    print("Wasting money? Fine, but you need to set openai_api = True first.")