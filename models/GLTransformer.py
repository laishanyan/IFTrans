"""
模型结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer
from .TextCNN import TextCNN
from .GTransformer import GTransformer
from .LTransformer import LTransformer


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)
        self.textcnn = TextCNN(config)
        self.gtransformer = GTransformer(config)
        self.ltransformer = LTransformer(config)
        self.device = config.device

    def embeding(self, x):
        context = x[0]  # 输入的句子
        supper = x[1]
        c_mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        s_mask = x[3]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_content, text_cls = self.bert(context, attention_mask=c_mask, output_all_encoded_layers=False)
        encoder_supper, text_cls = self.bert(supper, attention_mask=s_mask, output_all_encoded_layers=False)
        return encoder_content, encoder_supper

    def local_relevance(self, x1, x2):
        x1_out = self.textcnn(x1)
        x2_out = self.textcnn(x2)
        out = self.ltransformer(x1_out, x2_out)
        return out

    def glouble_relevance(self, x1, x2):
        out = self.gtransformer(x1, x2)
        return out

    def forward(self, x):
        encoder_x, encoder_supper = self.embeding(x)
        # 获取局部特征
        local = self.local_relevance(encoder_x, encoder_supper)
        glouble = self.glouble_relevance(encoder_x, encoder_supper)
        out = torch.cat([local, glouble], 1)
        out = self.fc(out)
        return out




