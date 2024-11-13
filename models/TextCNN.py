"""
TextCNN用于获取局部特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc_cnn = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_filters * len(config.filter_sizes))

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = x.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        return out