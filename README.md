<div align="center">

# Vector-Quantized Contrastive Predictive Coding <!-- omit in toc -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][notebook]
[![Paper](http://img.shields.io/badge/paper-arxiv.2005.09409-B31B1B.svg)][paper]  

</div>

Clone of official implmentation of VQ-CPC+Vocoder for [voice conversion](https://ja.wikipedia.org/wiki/%E9%9F%B3%E5%A3%B0%E5%90%88%E6%88%90#%E9%9F%B3%E5%A3%B0%E5%A4%89%E6%8F%9B) & [acoustic unit discovery](https://ja.wikipedia.org/wiki/%E9%9F%B3%E5%A3%B0%E5%88%86%E6%9E%90#Acoustic_Unit_Discovery).  

<!-- generated by [Markdown All in One](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one) -->
- [Demo](#demo)
- [Quick Training](#quick-training)
- [How to Use](#how-to-use)
- [Results](#results)
- [Original paper](#original-paper)

<p align="center">
  <img width="784" height="340" alt="VQ-CPC model summary"
    src="https://raw.githubusercontent.com/bshall/VectorQuantizedCPC/master/model.png"><br>
  <sup><strong>Fig 1:</strong> VQ-CPC model architecture.</sup>
</p>

## Demo
- [Original repo's samples](https://bshall.github.io/VectorQuantizedCPC/)

## Quick Training
Jump to ☞ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][notebook], then Run. That's all!  

## How to Use

### Pretrained models <!-- omit in toc -->
Pretrained weights for the 2019 English and Indonesian datasets can be found [here](https://github.com/bshall/VectorQuantizedCPC/releases/tag/v0.1).

### 1. Install <!-- omit in toc -->
```bash
pip install -r requirements.txt
```

### 2. Data & Preprocessing <!-- omit in toc -->
"Batteries Included" 😉  
Dataset class transparently downloads ZeroSpeech2019 corpus and preprocesses it for you.  

### 3. Training <!-- omit in toc -->
Train VQ-CPC encoder, then train vocoder.  
Pre-trained weight of both models are [here](https://github.com/bshall/VectorQuantizedCPC/releases/tag/v0.1).  

#### 3-A. Encoder <!-- omit in toc -->
```bash
python train_cpc.py \
    data.dataset.adress_data_root=dataset/will_be/saved/here \
    checkpoint_dir=checkpoint/will_be/saved/here
```

#### 3-B. Vocoder <!-- omit in toc -->
```bash
python train_vocoder_main.py \
    data.dataset.adress_data_root=dataset/will_be/saved/here \
    cpc_checkpoint=checkpoints/cpc/english2019/model.ckpt-22000.pt \
```

Sample audios will be periodically generated in `./out_sample` directory.

### 4. Evaluation <!-- omit in toc -->

#### 4-A. Voice conversion <!-- omit in toc -->
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

#### 4-B. ABX Score <!-- omit in toc -->
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

## Results
### Training Speed <!-- omit in toc -->
- Encoder: 15.5 [iter/sec] @ NVIDIA P100 (AMP-) Google Colaboratory == 4.7 hours/22000epochs 
- Vocoder: 0.84 [iter/sec] @ NVIDIA      (AMP-) Google Colaboratory == 2.2 days/160000steps

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
[notebook]:https://colab.research.google.com/github/tarepan/VectorQuantizedCPC/blob/master/VQ_CPC_training.ipynb

## Note of Original Repository <!-- omit in toc -->
- Train data is ZR19/Development/unit (9474 utterances)

## Contact <!-- omit in toc -->
Please check [original repository](https://github.com/bshall/VectorQuantizedCPC).  
