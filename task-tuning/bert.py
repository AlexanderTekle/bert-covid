import torch
import torch.nn as nn
from transformers import BertModel

def max_pool(x):
    return x.max(2)[0]


def mean_pool(x, sl):
    return torch.sum(x, 1) / sl.unsqueeze(1).float()


class BertMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768, 2)

    def forward(self, x, sl=None):
        mask = (x != 0).float()
        emb, _ = self.bert(x, attention_mask=mask)  # [B, L, H_b]
        rep = emb[:, 0, :]  # [B, H_b]
        logits = self.fc(rep)  # [B, C]
        return logits
