<div align="center">

# Vector-Quantized Contrastive Predictive Coding <!-- omit in toc -->
<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][notebook] -->
[![Paper](http://img.shields.io/badge/paper-arxiv.2005.09409-B31B1B.svg)][paper]  

</div>

Clone of official implmentation of VQ-CPC+Vocoder for [voice conversion](https://ja.wikipedia.org/wiki/%E9%9F%B3%E5%A3%B0%E5%90%88%E6%88%90#%E9%9F%B3%E5%A3%B0%E5%A4%89%E6%8F%9B) & [acoustic unit discovery](https://ja.wikipedia.org/wiki/%E9%9F%B3%E5%A3%B0%E5%88%86%E6%9E%90#Acoustic_Unit_Discovery).  

<!-- generated by [Markdown All in One](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one) -->
- [Demo](#demo)
- [How to Use](#how-to-use)
    - [Requirements](#requirements)
    - [Install](#install)
    - [Training](#training)
    - [Evaluation](#evaluation)
- [Original paper](#original-paper)
- [Contact](#contact)

<p align="center">
  <img width="784" height="340" alt="VQ-CPC model summary"
    src="https://raw.githubusercontent.com/bshall/VectorQuantizedCPC/master/model.png"><br>
  <sup><strong>Fig 1:</strong> VQ-CPC model architecture.</sup>
</p>

This work is based on *CPC* and *original VQ-VAE*.

## Demo
- [Original repo's samples](https://bshall.github.io/VectorQuantizedCPC/)

## How to Use
<!-- ### Quick training <- omit in toc ->
Jump to **[Notebook in Google Colaboratory][notebook]**, then Run. that's all!!  
 -->
Pretrained weights for the 2019 English and Indonesian datasets can be found [here](https://github.com/bshall/VectorQuantizedCPC/releases/tag/v0.1).

### Requirements
- PyTorch>=1.4
- [NVIDIA/apex](https://github.com/NVIDIA/apex)

### Install
```bash
pip install -r requirements.txt
```

### Data and Preprocessing

1.  Download and extract the [ZeroSpeech2020 datasets](https://download.zerospeech.com/).

2.  Download the train/test splits [here](https://github.com/bshall/VectorQuantizedCPC/releases/tag/v0.1) 
    and extract in the root directory of the repo.
    
3.  Preprocess audio and extract train/test log-Mel spectrograms:
    ```
    python preprocess.py in_dir=/path/to/dataset dataset=[2019/english or 2019/surprise]
    ```
    Note: `in_dir` must be the path to the `2019` folder. 
    For `dataset` choose between `2019/english` or `2019/surprise`.
    Other datasets will be added in the future.
    
    Example usage:
    ```
    python preprocess.py in_dir=../datasets/2020/2019 dataset=2019/english
    ```
    
### Training
<!-- ### Training Speed <!- omit in toc ->
X3.37 [iter/sec] @ NVIDIA T4 Google Colaboratory (AMP+)
 -->
Train VQ-CPC model, then train Vocoder model.  
Pre-trained weight of both models are [here](https://github.com/bshall/VectorQuantizedCPC/releases/tag/v0.1).

#### VQ-CPC
```
python train_cpc.py checkpoint_dir=path/to/checkpoint_dir dataset=[2019/english or 2019/surprise]
```

Example usage:
```
python train_cpc.py checkpoint_dir=checkpoints/cpc/2019english dataset=2019/english
```

#### Vocoder
```
python train_vocoder.py cpc_checkpoint=path/to/cpc/checkpoint checkpoint_dir=path/to/checkpoint_dir dataset=[2019/english or 2019/surprise]
```

Example usage:
```
python train_vocoder.py cpc_checkpoint=checkpoints/cpc/english2019/model.ckpt-24000.pt checkpoint_dir=checkpoints/vocoder/english2019
```

### Evaluation

#### Voice conversion
Convert speaker identity of specified voices to specified speaker.

```
python convert.py cpc_checkpoint=path/to/cpc/checkpoint vocoder_checkpoint=path/to/vocoder/checkpoint in_dir=path/to/wavs out_dir=path/to/out_dir synthesis_list=path/to/synthesis_list dataset=[2019/english or 2019/surprise]
```

Note: the `synthesis list` is a `json` file:
```
[
    [
        "english/test/S002_0379088085", // the path (relative to `in_dir`) of the source `wav` files
        "V002", // the target speaker (see `datasets/2019/english/speakers.json` for a list of options)
        "V002_0379088085" // the target file name
    ]
]
```

Example usage:
```
python convert.py cpc_checkpoint=checkpoints/cpc/english2019/model.ckpt-25000.pt vocoder_checkpoint=checkpoints/vocoder/english2019/model.ckpt-150000.pt in_dir=../datasets/2020/2019 out_dir=submission/2019/english/test synthesis_list=datasets/2019/english/synthesis.json in_dir=../../Datasets/2020/2019 dataset=2019/english
```

#### ABX Score
For evaluation install [bootphon/zerospeech2020](https://github.com/bootphon/zerospeech2020).
    
1.  Encode test data for evaluation:
    ```
    python encode.py checkpoint=path/to/checkpoint out_dir=path/to/out_dir dataset=[2019/english or 2019/surprise]
    ```
    ```
    e.g. python encode.py checkpoint=checkpoints/2019english/model.ckpt-500000.pt out_dir=submission/2019/english/test dataset=2019/english
    ```
    
2. Run ABX evaluation script (see [bootphon/zerospeech2020](https://github.com/bootphon/zerospeech2020)).

You can preview ABX score of the pretrained english model in original repository; [link](https://github.com/bshall/VectorQuantizedCPC#abx-score)  

## Original paper
[![Paper](http://img.shields.io/badge/paper-arxiv.2005.09409-B31B1B.svg)][paper]  
<!-- https://arxiv2bibtex.org/?q=2005.09409&format=bibtex -->
```
@misc{2005.09409,
Author = {Benjamin van Niekerk and Leanne Nortje and Herman Kamper},
Title = {Vector-quantized neural networks for acoustic unit discovery in the ZeroSpeech 2020 challenge},
Year = {2020},
Eprint = {arXiv:2005.09409},
}
```

[paper]:https://arxiv.org/abs/2005.09409
<!-- [notebook]:https://colab.research.google.com/github/tarepan/Scyclone-PyTorch/blob/main/Scyclone_PyTorch.ipynb -->

## Contact
Please check [original repository](https://github.com/bshall/VectorQuantizedCPC).  
