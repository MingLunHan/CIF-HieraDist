# CIF-HieraDist

## Introduction
[INTERSPEECH 2023] Knowledge Transfer from Pre-trained Language Models to Cif-based Speech Recognizers via Hierarchical Distillation 

This repository is the official implementation for the hierarchical knowledge distillation (HieraDist) developed for CIF-based models. Please refer to the original paper for details: [Knowledge Transfer from Pre-trained Language Models to Cif-based Speech Recognizers via Hierarchical Distillation](https://arxiv.org/abs/2301.13003). 

## What can you do with this repository?

1. Train a CIF-based ASR model; 
2. Train a CIF-based ASR model with acoustic contrastive distillation (ACD);
3. Train a CIF-based ASR model with linguistic regression distillation (LRD);
4. Train a CIF-based ASR model with hierarchical knowledge distillation (HieraDist/HKD).
5. Conduct model inference.  

## Usage

### Installation

### Data preparation

### Model Training

### Model Inference

## Acknowledgments

This repository is developed based on [Fairseq](https://github.com/facebookresearch/fairseq). Thanks to Facebook AI Research for releasing the Fairseq framework.

## Citation

If you are inspired by this paper, use the core code of this repository, or conduct research related to it, please cite this paper with the following format:

```
@INPROCEEDINGS{han2023hieradist,
  author    = {Han, Minglun and Chen, Feilong and Shi, Jing and Xu, Shuang and Xu, Bo},
  title     = {Knowledge Transfer from Pre-trained Language Models to Cif-based Speech Recognizers via Hierarchical Distillation},
  booktitle = {{INTERSPEECH}},
  year      = {2023}
}
```
