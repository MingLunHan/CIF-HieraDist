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

You should install all dependecies with the following commands:
```
cd CIF-HieraDist
pip install -r requirements.txt
pip install -e ./
```

Let's take the AISHELL-1 dataset as an example and navigate to the corresponding working directory for this dataset:
```
cd egs/aishell1
```

### Data preparation

The development of this repository is based on the [Fairseq](https://github.com/facebookresearch/fairseq). Please refer to the original data preparation of [speech-to-text](https://github.com/facebookresearch/fairseq/tree/main/examples/speech_to_text) in Fairseq. You can also refer to the https://github.com/MingLunHan/CIF-HieraDist/blob/main/examples/speech_to_text/prep_aishell1_data.py and modify it for your datasets.

```
python ../../examples/speech_to_text/prep_aishell1_data.py --input-root ${YOUR_PATH_TO_AISHELL1} --output-root ./data/
```

Note that YOUR_PATH_TO_AISHELL1 is the parent directory of the AISHELL-1 dataset. 

### Model Training

To train a standard CIF-based ASR model, you should use the command:
```
bash run_train_aishell1_cif_small_exp35_14.sh
```

To train a CIF-based ASR model with HieraDist/HKD, you should first extract features from PLM with the following command:
```
bash run_extract_plm_feats.sh
```
The output json file of PLM features should be set in the configuration file in egs/aishell1/data. Then, you should use the command:
```
bash run_train_bert_distilled_cif_exp4_decdistill0p01_noscale_finalstate_contrastiveloss1p0_conttemp0p02_rmvrpt_neg700.sh
```

We provide the original training logs in [egs/aishell1](https://github.com/MingLunHan/CIF-HieraDist/tree/main/egs/aishell1) for comparison. 

### Model Inference

To conduct the inference for an ASR model, you should use the command:
```
bash run_infer.sh
```

We provide the original inference logs in [egs/aishell1](https://github.com/MingLunHan/CIF-HieraDist/tree/main/egs/aishell1) for comparison. 

## Key Results

When not using any extra language models, we can get the results in the following table:

| Methods | dev (CER \%) | test (CER \%) |
| --- | --- | --- |
| CIF | 4.5 | 4.9 |
| CIF + ACD | 4.2 | 4.7 |
| CIF + LRD | 4.0 | 4.5 |
| CIF + HieraDist | 3.8 | **4.2 (4.1 with better decoding hyper-parameters in later experiments)** |

With the language model trained with the text of AISHELL-1 itself, we can get:

| Methods | dev (CER \%) | test (CER \%) |
| --- | --- | --- |
| CIF | 4.4 | 4.8 |
| CIF + ACD | 4.2 | 4.6 |
| CIF + LRD | 4.0 | 4.4 |
| CIF + HieraDist | 3.8 | **4.1** |

## Acknowledgments

This repository is developed on [Fairseq](https://github.com/facebookresearch/fairseq). Thanks to the [Facebook AI Research](https://ai.facebook.com/) for releasing the Fairseq framework.

## Other Resources

- A work that connects speech and LLM (ChatGLM) with an all-neural inferface in an all-neural way: https://github.com/phellonchen/X-LLM & https://github.com/MingLunHan/X-LLM-Speech

- A PyTorch implementation of the independent CIF module: https://github.com/MingLunHan/CIF-PyTorch

- CIF-based Contextualization, Collaborative Decoding (ColDec): https://github.com/MingLunHan/CIF-ColDec

- CIF as a bridge to connect pre-trained acoustic models and pre-trained language models: https://github.com/aispeech-lab/w2v-cif-bert

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
