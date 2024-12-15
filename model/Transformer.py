import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, raw_src, src_mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, raw_src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return output

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, raw_src, src_mask=None, src_key_padding_mask=None):
        # Q, K from src, V from raw_src
        q = k = src
        v = raw_src

        src2 = self.self_attn(q, k, v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerBinaryClassifier(nn.Module):
    def __init__(self, config, device, hidden_size=128):
        super(TransformerBinaryClassifier, self).__init__()
        self.device = device
        self.input_size = config.input_dim

        self.encoder_layer = CustomTransformerEncoderLayer(d_model=self.input_size, nhead=config.n_heads,
                                                           dropout=config.drop_out, dim_feedforward=hidden_size)
        self.transformer_encoder = CustomTransformerEncoder(self.encoder_layer, num_layers=config.n_layer)

        self.bn = nn.BatchNorm1d(self.input_size)
        self.dropout = nn.Dropout(config.drop_out)
        self.classifier = nn.Linear(self.input_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.to(device)
        self.config = config

    def forward(self, raw_input, GNN_output, mask):
        GNN_output = GNN_output.transpose(0, 1)
        raw_input = raw_input.transpose(0, 1)
        GNN_output = GNN_output.float()
        raw_input = raw_input.float()
        mask = mask.bool()

        transformer_output = self.transformer_encoder(GNN_output, raw_input, src_key_padding_mask=mask)
        transformer_output = transformer_output.transpose(0, 1)
        transformer_output[mask] = 0

        filtered_output = [out[~m] for out, m in zip(transformer_output, mask)]

        averaged_tensors = [tensor.mean(dim=0) for tensor in filtered_output]

        mean_output = torch.stack(averaged_tensors)

        mean_output = self.bn(mean_output)

        logits = self.classifier(mean_output)
        logits = self.sigmoid(logits).squeeze()

        return logits
