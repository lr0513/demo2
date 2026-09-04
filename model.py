import torch.nn as nn
from transformers import AutoModel


class BertNERModel(nn.Module):
    def __init__(self, pretrain_name: str, hidden_size: int, num_labels: int, dropout: float):
        super(BertNERModel, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrain_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        # 训练阶段
        if labels is not None:
            # 忽略标签为-100的位置（[CLS]、[SEP]、padding）
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # 展平计算loss
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {
            "loss": loss,
            "logits": logits
        }
