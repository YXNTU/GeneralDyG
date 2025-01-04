# A Generalizable Anomaly Detection Method in Dynamic Graphs
This repository contains the official implementation of the paper: ["A Generalizable Anomaly Detection Method in Dynamic Graphs"](https://arxiv.org/abs/2412.16447), accepted at AAAI 2025.
## Abstract
Anomaly detection aims to identify deviations from normal patterns within data. This task is particularly crucial in dynamic graphs, which are common in applications like social networks and cybersecurity, due to their evolving structures and complex relationships. Although recent deep learning based methods have shown promising results in anomaly detection on dynamic graphs, they often lack of generalizability. In this study, we propose GeneralDyG, a method that samples temporal ego-graphs and sequentially extracts structural and temporal features to address the three key challenges in achieving generalizability: Data Diversity, Dynamic Feature Capture, and Computational Cost. Extensive experimental results demonstrate that our proposed GeneralDyG significantly outperforms state-of-the-art methods on four real world datasets.
![framework](./process.png)

## Requirements

![h5py](https://img.shields.io/badge/h5py-3.7.0-blue)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.12.3-orange)
![imblearn](https://img.shields.io/badge/imblearn-0.0-lightgrey)
![matplotlib](https://img.shields.io/badge/matplotlib-3.10.0-purple)
![networkx](https://img.shields.io/badge/networkx-2.8.7-darkblue)
![numpy](https://img.shields.io/badge/numpy-1.23.3-yellow)
![pandas](https://img.shields.io/badge/pandas-1.4.4-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.0-red)
![scipy](https://img.shields.io/badge/scipy-1.8.1-cyan)
![torch](https://img.shields.io/badge/torch-2.1.2%2Bcu121-brightgreen)
![torch-geometric](https://img.shields.io/badge/torch--geometric-2.2.0-lightblue)
![torch-scatter](https://img.shields.io/badge/torch--scatter-2.1.0%2Bpt112cu116-lightgreen)
![torch-sparse](https://img.shields.io/badge/torch--sparse-0.6.18-gold)
![tqdm](https://img.shields.io/badge/tqdm-4.65.2-pink)

## Preprocessing
Here, we provide two preprocessed datasets: **Bitcoin-Alpha** and **Bitcoin-OTC**. Please download the preprocessed datasets [download the dataset](https://drive.google.com/drive/folders/1nJGwX0QaWZY3RH8JfqogJYMbq9PXkYhC?usp=sharing) and extract them into the current directory.

You can choose to preprocess the data before training or use the two sample files we provided. Please run the following command to preprocess the data:

```bash
python generate_datasets.py
```

### Instructions
- In `generate_datasets.py`, you can adjust the parameters `k` and `dataset_name` to generate different versions of preprocessed data.
- **`k`**: Controls specific preprocessing behaviors.
- **`dataset_name`**: Specifies the dataset to preprocess.

### Provided Preprocessed Data
We provide preprocessed versions of the **Alpha** and **OTC** datasets with `k=1`.  
These preprocessed datasets can be found in the `dataset/` directory.

### Directory Structure
```plaintext
dataset/
├── btc_alpha.pkl/
└── btc_otc.pkl/
```

## Start Training

After completing the preprocessing step, start the training process by running:

```bash
python train.py
```


