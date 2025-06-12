import torch
from trl import SFTConfig, SFTTrainer
from trl.trainer import ConstantLengthDataset
from datasets import load_dataset, Dataset
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from peft import get_peft_model, LoraConfig, PeftModel, PeftConfig, AutoPeftModelForCausalLM
from huggingface_hub import login
from utils import base_dependencies
import pandas as pd

import json
import os
import shutil
import evaluate
from lens import download_model, LENS
import textstat
from langdetect import detect_langs
from langdetect import DetectorFactory
import re

################################################################################ SAVE FUNCTION

def save_score(eval_directory, prefix, subdirectory, model_name, score, dev_or_test = 'dev'):
    if dev_or_test == 'dev':
        eval_directory = "outputs/sft_dev_eval/"
        with open(f"{eval_directory}{prefix}_{subdirectory[:-1]+'_'}{model_name}.txt", "w", encoding="utf-8") as file:
            json.dump(score, file, ensure_ascii=False, indent=4)
    elif dev_or_test == 'test':
        eval_directory = "outputs/sft_test_eval/"
        fname = f"{eval_directory}{prefix}_{subdirectory[:-1]+'_'}{model_name}.txt"
        fname = fname.replace('bs16ga1dv1lr1e-4/','bs16ga1dv1lr1e-4_')
        with open(fname, "w", encoding="utf-8") as file:
            json.dump(score, file, ensure_ascii=False, indent=4)

################################################################################ GENERATE OUTPUT FOR TEST DATASET

def get_generations(model_name, test_dataset, subdirectory, purpose, max_seq_length = 256, subdirectory_fixer = False):

    if subdirectory_fixer == True:
        subdirectory_model_fix = subdirectory.replace('test/', '')
    else:
        subdirectory_model_fix = subdirectory
    if not os.path.exists(f'outputs/generations/{subdirectory}'):
        os.makedirs(f'outputs/generations/{subdirectory}')

    generations_exist = os.path.exists(f'outputs/generations/{subdirectory}' + model_name + ".json")
    
    print(f'outputs/generations/{subdirectory}', generations_exist)
    if not generations_exist:

        print(f'getting generations for {subdirectory}{model_name}')
        # import model + tokenizer
        model, tokenizer = base_dependencies.load_model_and_tokenizer_wrapper(subdirectory_model_fix + model_name, purpose)
        print(f"Model on GPU? {next(model.parameters()).is_cuda}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        base_dependencies.generate_output(model, model_name, tokenizer, test_dataset, subdirectory, max_seq_length, save = True)

    return base_dependencies.load_jsonl(f'outputs/generations/{subdirectory}' + model_name + '.json')[0]

################################################################################ SLATE OF EVALS

def eval_slate(eval_type, model_name, orig_sents, ref_sents, gen_sents, test_dataset, subdirectory, max_seq_length, dev_or_test, ignore_equals = False):

    print(dev_or_test, orig_sents[0], gen_sents[0])
    # removes minor irrelevant elements leftover (i assume) from mishandled chat templates
    gen_sents = [base_dependencies.clean_up_chat_template_leftovers(s) for s in gen_sents]

    orig_sents_copy = orig_sents
    gen_sents_copy = gen_sents

    if ignore_equals == True:
        tmp = pd.DataFrame({'orig': orig_sents, 'ref': ref_sents, 'gen': gen_sents})
        total_sent = len(tmp)
        tmp = tmp[tmp['orig'] != tmp['gen']]
        orig_sents = list(tmp['orig'])
        ref_sents = list(tmp['ref'])
        gen_sents = list(tmp['gen'])
        print('dropped', total_sent - len(tmp), 'sentences from consideration')

    if dev_or_test == 'test':
        subdirectory_model_fix = subdirectory.replace('test/', '')
    else:
        subdirectory_model_fix = subdirectory

    # perform evaluation
    if eval_type == "bert" or eval_type == "all":
        run_evalbert(model_name, orig_sents, gen_sents, subdirectory, dev_or_test, save = True)
    if eval_type == "bleu" or eval_type == "all":
        run_evalbleu(model_name, orig_sents, gen_sents, subdirectory, dev_or_test, save = True)
    if eval_type == "ease" or eval_type == "all":
        run_evalease(model_name, gen_sents, subdirectory, dev_or_test, save = True)
    if eval_type == "empt" or eval_type == "all":
        run_evalempt(model_name, gen_sents, subdirectory, dev_or_test, save = True)
    if eval_type == "equl" or eval_type == "all":
        run_evalequl(model_name, orig_sents_copy, gen_sents_copy, subdirectory, dev_or_test, save = True)
    if eval_type == "lens": # or eval_type == "all":
        run_evallens(model_name, orig_sents, ref_sents, gen_sents, subdirectory, dev_or_test, save = True)
    if eval_type == "lang" or eval_type == "all":
        run_evallang(model_name, gen_sents, subdirectory, dev_or_test, save = True)
    if eval_type == "leng" or eval_type == "all":
        run_evalleng(model_name, gen_sents, subdirectory, dev_or_test, save = True)
    #if eval_type == "loss" or eval_type == "all":
    #    model, tokenizer = base_dependencies.load_model_and_tokenizer_wrapper(subdirectory_model_fix + model_name, 'sft_train')
    #    run_evalloss_sfttrainer(model, model_name, tokenizer, test_dataset, subdirectory, max_seq_length, dev_or_test, save = True)
    if eval_type == "sari" or eval_type == "all":
        run_evalsari(model_name, orig_sents, ref_sents, gen_sents, subdirectory, dev_or_test, save = True)
    if eval_type == "wstf" or eval_type == "all":
        run_evalwstf(model_name, gen_sents, subdirectory, dev_or_test, save = True)

################################################################################ LOSS EVALUATION VIA SFTTRAINER

def run_evalloss_sfttrainer(model, model_name, tokenizer, test_dataset, subdirectory, max_seq_length, dev_or_test, save = False):

    training_args = SFTConfig(
        output_dir="outputs",
        max_seq_length = max_seq_length,
        logging_dir=None,
        logging_strategy="no"
    )

    #trainer = SFTTrainer(
    #    model = model,
    #    tokenizer = tokenizer,
    #    train_dataset = test_dataset,
    #    eval_dataset = test_dataset,
    #    args = training_args,
    #)

    if 'llama' in model_name.lower(): #and 'disco' not in model_name.lower()
        print('for completion only')
        trainer = SFTTrainer(
            model = model, 
            tokenizer = tokenizer,
            eval_dataset = test_dataset,
            args = training_args,
            formatting_func = base_dependencies.formatting_prompts_func,
            data_collator = base_dependencies.load_collator(tokenizer),
        )
    else:
        print('next token pred')
        trainer = SFTTrainer(
            model = model, 
            tokenizer = tokenizer,
            eval_dataset = test_dataset,
            args = training_args,
        )

    print(test_dataset)
    torch.cuda.empty_cache()
    evaluation_stats = trainer.evaluate()
    test_loss = evaluation_stats['eval_loss']

    if save == True:
        save_score(dev_or_test, 'loss', subdirectory, model_name, test_loss, dev_or_test)

################################################################################ HUGGINGFACE EVALUATIONS

def run_evalbleu(model_name, orig_sents, gen_sents, subdirectory, dev_or_test, save = False):
    bleu = evaluate.load("bleu")
    bleu_results = bleu.compute(predictions=gen_sents, references=[[orig_sent] for orig_sent in orig_sents])
    if save == True:
        save_score(dev_or_test, "bleu", subdirectory, model_name, bleu_results, dev_or_test)

def run_evalbert(model_name, orig_sents, gen_sents, subdirectory, dev_or_test, save = False):
    bertscore = evaluate.load("bertscore")
    bert_results = bertscore.compute(references = orig_sents, predictions = gen_sents, lang="de")
    if save == True:
        save_score(dev_or_test, "bert", subdirectory, model_name, bert_results, dev_or_test)

def run_evalsari(model_name, orig_sents, ref_sents, gen_sents, subdirectory, dev_or_test, save = False):
    sari = evaluate.load("sari")
    sari_value = sari.compute(sources = orig_sents, predictions = gen_sents, references = [[ref_sent] for ref_sent in ref_sents])
    sari_value2 = [sari.compute(sources = [orig_sent], predictions = [gen_sent], references = [[refs]])['sari'] for orig_sent, gen_sent, refs in zip(orig_sents, gen_sents, ref_sents)]
    if save == True:
        save_score(dev_or_test, "sari", subdirectory, model_name, (sari_value, sari_value2), dev_or_test)

################################################################################ LENS EVALUATIONS

def run_evallens(model_name, orig_sents, ref_sents, gen_sents, subdirectory, dev_or_test, save = False):
    lens_path = download_model("davidheineman/lens")
    lens = LENS(lens_path, rescale=True)
    scores = lens.score(orig_sents, gen_sents, [[ref_sent] for ref_sent in ref_sents], batch_size=8, devices=[0])
    if save == True:
        save_score(dev_or_test, "lens", subdirectory, model_name, scores, dev_or_test)

################################################################################ READABILITY EVALUATIONS

def run_evalease(model_name, gen_sents, subdirectory, dev_or_test, save = False):
    textstat.set_lang('de')
    scores = [textstat.flesch_reading_ease(sent) for sent in gen_sents]
    if save == True:
        save_score(dev_or_test, "ease", subdirectory, model_name, scores, dev_or_test)

def run_evalwstf(model_name, gen_sents, subdirectory, dev_or_test, save = False):
    textstat.set_lang('de')
    scores = []
    i = 0
    for sent in gen_sents:
        try:
            scores.append(textstat.wiener_sachtextformel(sent, 4))
        except:
            i += 1
            print(i, 'generations do not include words!')
            pass
    if save == True:
        save_score(dev_or_test, "wstf", subdirectory, model_name, scores, dev_or_test)

################################################################################ OUTPUT LENGTH

def run_evalempt(model_name, gen_sents, subdirectory, dev_or_test, save = False):
    lengths = [len(sent) for sent in gen_sents]
    blanks = [length for length in lengths if length < 10]
    prop_blank = len(blanks) / len(lengths)
    if save == True:
        save_score(dev_or_test, 'empt', subdirectory, model_name, prop_blank, dev_or_test)

def run_evalequl(model_name, orig_sents, gen_sents, subdirectory, dev_or_test, save = False):

    #num_equal = 0
    #for i in range(len(orig_sents)):
    #    if re.sub(r'[^a-zA-Z]', '', orig_sents[i]).lower() == re.sub(r'[^a-zA-Z]', '', gen_sents[i]).lower():
    #        num_equal += 1
    #        print(orig_sents[i], gen_sents[i])
    
    orig_clean = [re.sub(r'[^a-zA-Z]', '', sent).lower() for sent in orig_sents]
    gen_clean = [re.sub(r'[^a-zA-Z]', '', sent).lower() for sent in gen_sents]
    common_sentences = [1 for i in range(len(gen_sents)) if gen_clean[i] == orig_clean[i]]

    if save == True:
        save_score(dev_or_test, 'equl', subdirectory, model_name, len(common_sentences) / len(gen_clean), dev_or_test)

def run_evalleng(model_name, gen_sents, subdirectory, dev_or_test, save = False):
    wcs = (base_dependencies.wcs(gen_sents))
    avg_wc = sum(wcs) / len(wcs)
    if save == True:
        save_score(dev_or_test, 'leng', subdirectory, model_name, avg_wc, dev_or_test)

################################################################################ SHARE OUTPUT GERMAN

def detectlangs_wrapper(s):
    try:
        return detect_langs(s)
    except:
        return 'no detected lang'

def firstlang_wrapper(langdet):
    if langdet == 'no detected lang':
        return {'??': 1}
    else:
        return {langdet[0].lang: langdet[0].prob}
    
def run_evallang(model_name, gen_sents, subdirectory, dev_or_test, save = False):
    DetectorFactory.seed = 5
    lang_detects = [detectlangs_wrapper(s) for s in gen_sents]
    first_lang_guess = [firstlang_wrapper(langdet) for langdet in lang_detects]
    de_count = 0
    for guess in first_lang_guess:
        if 'de' in guess.keys():
            de_count += 1
    score = de_count / len(first_lang_guess)
    if save == True:
        save_score(dev_or_test, "lang", subdirectory, model_name, score, dev_or_test)