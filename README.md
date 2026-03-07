# AAD-LLM: Neural Attention-Driven Auditory Scene Understanding

[![arXiv](https://img.shields.io/badge/arXiv-2502.16794-b31b1b.svg)](https://arxiv.org/abs/2502.16794)
[![Website](https://img.shields.io/badge/Website-Demo-blue)](https://aad-llm.github.io)

Official implementation of **AAD-LLM: Neural Attention-Driven Auditory Scene Understanding**.

---

## Overview

**AAD-LLM** is both an audio understanding model and a brain-computer interface that can selectively answer your question according to the listener's auditory attention from the speech mixture.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b3dbfba8-efc1-4dde-b149-fd55c2f84ee8" width="53%" />
  <img src="https://github.com/user-attachments/assets/263e54ca-0c3e-44af-a098-1ebe9ca02d34" width="42%" />
</p>

---

## Prerequisites

Install necessary packages (including those required by [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B).

```code
conda create -n AADLLM python=3.11
conda activate AADLLM
pip install -r requirements.txt
```
Then, go to the source code of the transformers package you just installed. Find the file ***transformers/models/qwen2_audio/modeling_qwen2_audio.py***. Make two changes in Qwen2AudioEncoder.

```code
1. Removed the check input_features.shape[-1] != expected_seq_length
2. hidden_states = inputs_embeds + embed_pos -> hidden_states = inputs_embeds + embed_pos[0:inputs_embeds.shape[1]]
```

---

## Checkpoints and Test Data

The AAD-LLM checkpoint based on Qwen2-Audio and trained on K-Means of x-vectors (K=8) is [here](https://github.com/xi-j/AAD-LLM).

samples/ folder contains public test mixtures of speeches from [LibriTTS](https://www.openslr.org/60) and background noise from [DEMAND](https://zenodo.org/records/1227121). 

We apologize that we were unable to open-source clinical recordings and brain data.

---

## Inference

Please check **inference_Qwen2AADLLM.ipynb** for offline inference. We use the ground-truth x-vector label of the attended speaker to condition LLM's generation. To make it a BCI application, the x-vector label of the attended should be predicted from the brain (with your choice of the brain signal).

```code
conda install -c conda-forge jupyterlab
```

---

## Citation

If you find this work helpful, please consider to cite
```text
@inproceedings{jiang-etal-2025-aad,
    title = "{AAD}-{LLM}: Neural Attention-Driven Auditory Scene Understanding",
    author = "Jiang, Xilin  and
      Dindar, Sukru Samet  and
      Choudhari, Vishal  and
      Bickel, Stephan  and
      Mehta, Ashesh  and
      McKhann, Guy M  and
      Friedman, Daniel  and
      Flinker, Adeen  and
      Mesgarani, Nima",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1257/",
    doi = "10.18653/v1/2025.acl-long.1257",
    pages = "25887--25909",
    ISBN = "979-8-89176-251-0",
}
```

