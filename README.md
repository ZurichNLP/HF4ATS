# HF4ATS: Human Feedback for Automatic Text Simplification

This repository contains the code for the data processing, web application, model training, experiments, and analyses described in the paper ["Evaluating the Effectiveness of Direct Preference Optimization for Personalizing German Automatic Text Simplifications for Persons with Intellectual Disabilities"]().

All code and data is intended for non-commercial research use only – please refer to the accompanying license/copyright notice for details.

## HF4ATS Data

HF4ATS (**H**uman **F**eedback **For** **A**utomatic **T**ext **S**implification) is a German-language collection of news sentence preferences and simplifications suitable for simplification preference alignment and/or fine-tuning. It is composed of:
1. <u>HF4ATS-DPO</u> Automatic text simplification (ATS) preference pairs annotated by both text simplification experts and persons with cognitive impairments. This dataset is suitable for preference alignment.
2. <u>HF4ATS-SFT</u> Complex-simple manual text simplification pairs. This dataset is suitable for supervised fine-tuning.

HF4ATS-DPO data is available at [Zenodo].
HF4ATS-SFT data is available as `data/sft_<type>.jsonl`, where type can be `train`, `dev`, and `holdout`. This data is sourced from DEplain-APA data, available on [Zenodo](https://zenodo.org/records/8304430). 

## Usage

Requirements:

* Python == 3.9

```bash
pip install -r requirements.txt
```

### Preparing DEPlain-APA data for Supervised Fine-Tuning

Train, development, and test HF4ATS-SFT data is available already in `data/`. That said, it can be reproduced from the raw DEplain-APA data. 

First, download the [DEPlain-APA](https://zenodo.org/records/8304430) data. Place `all.csv` inside `data/deplain_sentences/`.

Next, run the following script:

```bash
python pre_sft.py
```

At this point, the complex-simple SFT data is ready for use under `data/sft_train.jsonl`, `data/sft_dev.jsonl`, and `data/sft_holdout.jsonl`.

### Supervised Fine-Tuning

Our model checkpoints are available at [swissubase]. 

Should you wish to perform SFT, you must first ensure you have access (if required) to the following four models on HuggingFace:

https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
https://huggingface.co/DiscoResearch/Llama3-DiscoLeo-Instruct-8B-v0.1 
https://huggingface.co/LeoLM/leo-mistral-hessianai-7b-chat

Next, input your HuggingFace token in place of the text `<YOUR HUGGINGFACE TOKEN HERE>` inside `utils/base_dependencies.py`

Finally, run the following script:

```bash
python sft_path.py --args
```

### Direct Preference Optimization

### Automatic Evaluation Metrics for an SFT or DPO Checkpoint

## Copyright notice

The resulting dataset is released with the following copyright notice:

### German / Deutsch:

### English / Englisch:

## Citation

If you use the dataset, please cite the following paper:

```
@misc{ 
}
```




