# A Generalizable Anomaly Detection Method in Dynamic Graphs
This repository contains the official implementation of the paper: ["A Generalizable Anomaly Detection Method in Dynamic Graphs"](https://arxiv.org/abs/2412.16447), accepted at AAAI 2025.
## Requirements

- **h5py**: `3.7.0`  
- **imbalanced-learn**: `0.12.3`  
- **imblearn**: `0.0`  
- **matplotlib**: `3.10.0`  
- **networkx**: `2.8.7`  
- **numpy**: `1.23.3`  
- **pandas**: `1.4.4`  
- **scikit-learn**: `1.6.0`  
- **scipy**: `1.8.1`  
- **torch**: `2.1.2+cu121`  
- **torch-geometric**: `2.2.0`  
- **torch-scatter**: `2.1.0+pt112cu116`  
- **torch-sparse**: `0.6.18`  
- **tqdm**: `4.65.2`  

## Preprocessing

[download the dataset](https://drive.google.com/drive/folders/1nJGwX0QaWZY3RH8JfqogJYMbq9PXkYhC?usp=sharing)

Before running the main code, generate the necessary preprocessed data files by executing:

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

## Model Structure
[process.pdf](https://github.com/user-attachments/files/18140405/process.pdf)

