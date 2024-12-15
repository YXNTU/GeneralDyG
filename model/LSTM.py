import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMBinaryClassifier(nn.Module):
    def __init__(self, config, device, hidden_size=128):
        super(LSTMBinaryClassifier, self).__init__()
        self.device = device
        self.input_size = config.input_dim
        self.hidden_size = hidden_size
        self.num_layers = config.n_layer

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True, dropout=config.drop_out)

        self.bn = nn.BatchNorm1d(self.hidden_size)
        self.dropout = nn.Dropout(config.drop_out)
        self.classifier = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.to(device)

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.float()
        attention_mask_bool = attention_mask.bool()

        # 通过LSTM层
        lstm_output, (hn, cn) = self.lstm(input_ids)

        # 经过BatchNorm
        lstm_output = self.bn(lstm_output.transpose(1, 2)).transpose(1, 2)

        # 获取最后一个时间步的输出作为logits
        # 根据 attention_mask 过滤出最后一个有效时间步的输出
        lengths = (~attention_mask_bool).sum(dim=1)
        last_output = lstm_output[range(len(lstm_output)), lengths - 1, :]

        logits = self.classifier(last_output)
        logits = self.sigmoid(logits).squeeze()

        # 使用 attention_mask 过滤出原有的部分
        filtered_output = [out[~m] for out, m in zip(lstm_output, attention_mask_bool)]

        return {
            'logits': logits,
            'filtered_output': filtered_output
        }
