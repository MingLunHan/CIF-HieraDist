# CIF-HieraDist

## Introduction
[INTERSPEECH 2023] Knowledge Transfer from Pre-trained Language Models to Cif-based Speech Recognizers via Hierarchical Distillation 

This repository is the official implementation for the hierarchical knowledge distillation (HieraDist) developed for CIF-based models. Please refer to the original paper for more details: [Knowledge Transfer from Pre-trained Language Models to Cif-based Speech Recognizers via Hierarchical Distillation](https://arxiv.org/abs/2301.13003). 

## What can you do with this repository?

1. Train a CIF-based ASR model; 
2. Train a CIF-based ASR model with acoustic contrastive distillation (ACD);
3. Train a CIF-based ASR model with linguistic regression distillation (LRD);
4. Train a CIF-based ASR model with hierarchical knowledge distillation (HieraDist/HKD).
5. Conduct model inference.  

## Usage

### Installation

My default python version:
```
python==3.7.9
```

You should install all dependecies with following commands:
```
cd CIF-HieraDist
pip install -r requirements.txt
pip install -e ./
```

### Data preparation

The development of this repository is based on the [Fairseq](https://github.com/facebookresearch/fairseq). Please refer to the original data preparation of [speech-to-text](https://github.com/facebookresearch/fairseq/tree/main/examples/speech_to_text) in Fairseq. You can also refer to the https://github.com/MingLunHan/CIF-HieraDist/blob/main/examples/speech_to_text/prep_aishell1_data.py and modify it for your datasets.

### Model Training

### Model Inference

## Acknowledgments

This repository is developed based on [Fairseq](https://github.com/facebookresearch/fairseq). Thanks to the [Facebook AI Research](https://ai.facebook.com/) for releasing the Fairseq framework.

## Citation

If you are inspired by this paper, or use the core codes from this repository for your development, or conduct research related to it, please cite this paper with the following bibtex format:

```
@INPROCEEDINGS{han2023hieradist,
  author    = {Han, Minglun and Chen, Feilong and Shi, Jing and Xu, Shuang and Xu, Bo},
  title     = {Knowledge Transfer from Pre-trained Language Models to Cif-based Speech Recognizers via Hierarchical Distillation},
  booktitle = {{INTERSPEECH}},
  year      = {2023}
}
```

Thanks!
