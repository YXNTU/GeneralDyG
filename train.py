import sys

import torch
import torch.nn.functional as F
import datasets as dataset
import torch.utils.data
import sklearn
from scipy.stats import rankdata, iqr, trim_mean
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
import numpy as np
from torch_sparse import SparseTensor

from model.CensNet import CensNet
from model.Combine import CombinedModel
from model.LSTM import LSTMBinaryClassifier
from option import args
from utils import EarlyStopMonitor, logger_config
from tqdm import tqdm
import datetime, os
from model.Transformer import TransformerBinaryClassifier
from imblearn.over_sampling import RandomOverSampler
# import nni
import random


def main():
    def criterion(logits, labels):
        labels = labels
        loss_classify = F.binary_cross_entropy_with_logits(
            logits, labels, reduction='none')
        loss_classify = torch.mean(loss_classify)

        return loss_classify

    def eval_epoch(dataset, model, config, device):
        m_loss, m_pred, m_label = np.array([]), np.array([]), np.array([])
        with torch.no_grad():
            model.eval()
            for batch_sample in dataset:
                input_nodes_feature = batch_sample['input_nodes_feature']
                input_edges_feature = batch_sample['input_edges_feature']
                input_edges_pad = batch_sample['input_edges_pad']
                labels = batch_sample['labels']
                Tmats = batch_sample['Tmats']
                adjs = batch_sample['adjs']
                eadjs = batch_sample['eadjs']
                mask_edge = batch_sample['mask_edge']

                input_nodes_feature = [tensor.to(device) for tensor in input_nodes_feature]
                input_edges_feature = [tensor.to(device) for tensor in input_edges_feature]
                Tmats = [tensor.to(device) for tensor in Tmats]
                adjs = [tensor.to(device) for tensor in adjs]
                eadjs = [tensor.to(device) for tensor in eadjs]

                logits = model(
                    input_nodes_feature,
                    input_edges_feature,
                    input_edges_pad.to(device),
                    eadjs,
                    adjs,
                    Tmats,
                    mask_edge.to(device)
                )
                y = labels.to(device)
                y = y.to(torch.float32)

                c_loss = np.array([criterion(logits, y).cpu()])
                pred_score = logits.cpu().numpy().flatten()
                y = y.cpu().numpy().flatten()
                m_loss = np.concatenate((m_loss, c_loss))
                m_pred = np.concatenate((m_pred, pred_score))
                m_label = np.concatenate((m_label, y))
            auc_roc = roc_auc_score(m_label, m_pred)
        return np.mean(m_loss), auc_roc

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    config = args
    params_nni = {
        'seed': config.seed,
        'batch_size': 32,
        'w_time': 0.01,
        'n_heads': 4,
        'drop_out': 0.3,
        'n_layer': 6,
        'task': 'Node'
    }

    set_seed(params_nni['seed'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    max_num = [0]

    dataset_train = dataset.DygDataset(config, 'train')
    dataset_test = dataset.DygDataset(config, 'test')

    gpus = None if config.gpus == 0 else config.gpus

    collate_fn = dataset.Collate(config)

    GNN = CensNet(config.input_dim, config.drop_out)
    transformer = TransformerBinaryClassifier(config, device, hidden_size=config.hidden_dim)

    backbone = CombinedModel(GNN, transformer)

    model = backbone.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    loader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_data_workers,
        pin_memory=True,
        collate_fn=collate_fn.dyg_collate_fn
    )

    loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_data_workers,
        collate_fn=collate_fn.dyg_collate_fn
    )

    max_test_auc = 0.0
    early_stopper = EarlyStopMonitor(higher_better=True)
    # count = 0
    for epoch in range(config.n_epochs):
        auc = []
        with tqdm(total=len(loader_train)) as t:
            for batch_idx, batch_sample in enumerate(loader_train):
                input_nodes_feature = batch_sample['input_nodes_feature']
                input_edges_feature = batch_sample['input_edges_feature']
                input_edges_pad = batch_sample['input_edges_pad']
                labels = batch_sample['labels']
                Tmats = batch_sample['Tmats']
                adjs = batch_sample['adjs']
                eadjs = batch_sample['eadjs']
                mask_edge = batch_sample['mask_edge']

                input_nodes_feature = [tensor.to(device) for tensor in input_nodes_feature]
                input_edges_feature = [tensor.to(device) for tensor in input_edges_feature]
                Tmats = [tensor.to(device) for tensor in Tmats]
                adjs = [tensor.to(device) for tensor in adjs]
                eadjs = [tensor.to(device) for tensor in eadjs]

                t.set_description('Epoch %i' % epoch)
                optimizer.zero_grad()
                model.train()
                logits = model(
                    input_nodes_feature,
                    input_edges_feature,
                    input_edges_pad.to(device),
                    eadjs,
                    adjs,
                    Tmats,
                    mask_edge.to(device)
                )
                y = labels.to(device)
                y = y.to(torch.float32)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                t.set_postfix(loss=loss)
                t.update(1)

        test_loss, test_auc = eval_epoch(loader_test, model, config, device)
        print(f' val loss in epoch: {test_loss}')
        print(f' val auc in epoch: {test_auc}')
        if test_auc > max_test_auc:
            max_test_auc = test_auc
        if early_stopper.early_stop_check(test_auc):  # 注意这里得变一下
            print('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
            break
        else:
            pass

    print('best auc:{}'.format(max_test_auc))

if __name__ == "__main__":
    main()

