# HF4ATS: Human Feedback for Automatic Text Simplification

This repository contains the code for the data processing, web application, model training, experiments, and analyses described in the paper ["Evaluating the Effectiveness of Direct Preference Optimization for Personalizing German Automatic Text Simplifications for Persons with Intellectual Disabilities"]().

All code and data is intended for non-commercial research use only – please refer to the accompanying license/copyright notice for details.

## HF4ATS Data

HF4ATS (**H**uman **F**eedback **For** **A**utomatic **T**ext **S**implification) is a German-language collection of news sentence preferences and simplifications suitable for simplification preference alignment and/or fine-tuning. It is composed of:
1. <u>HF4ATS-DPO</u> Automatic text simplification (ATS) preference pairs annotated by both text simplification experts and persons with cognitive impairments. This dataset is suitable for preference alignment.
2. <u>HF4ATS-SFT</u> Complex-simple manual text simplification pairs. This dataset is suitable for supervised fine-tuning.

HF4ATS-DPO data is available at [Zenodo].
HF4ATS-SFT data is processed from DEPlain-APA data, available on [Zenodo](https://zenodo.org/records/8304430). 


## Usage

Requirements:

* Python == 3.11

```bash
pip install -r reqs.txt
```

### Preparing DEPlain-APA data for Supervised Fine-Tuning

```bash
python fp/file.py
```

At this point, the sentence-level data is ready for use under `fp/data.tsv`.

### Supervised Fine-Tuning

```bash
python fp/file.py --args
```

```bash
python scripts/clean_scraped_dataset.py --drop-dates
```

### Automatic Evaluation Metrics for an SFT Checkpoint

### Direct Preference Optimization

### Automatic Evaluation Metrics for a DPO Checkpoint

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




