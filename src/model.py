import os
from dataclasses import asdict

import torch
import torch.nn as nn
from transformers import AutoModel

from src.utils import mkdir_if_not_exist


class BertNERModel(nn.Module):
    def __init__(self, pretrain_name: str, num_labels: int, dropout: float):
        super(BertNERModel, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrain_name)
        hidden_size = self.bert.config.hidden_size
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
            # 忽略标签为-100的位置（[CLS]、[SEP]、padding、子词）
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # 展平计算loss
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {
            "loss": loss,
            "logits": logits
        }


def save_model_artifact(path: str, model: BertNERModel, label_map: dict, cfg) -> str:
    """把模型权重、标签映射和实验配置保存到同一个文件中。"""
    parent_dir = os.path.dirname(path)
    mkdir_if_not_exist(parent_dir)

    artifact = {
        "model_state_dict": model.state_dict(), # 神经网络每一层所有训练好的权重数字
        "label_map": label_map,
        "pretrain_name": cfg.model.pretrain_name,
        "dropout": cfg.train.dropout,
        "hidden_size": model.bert.config.hidden_size,
        "max_len": cfg.train.max_len,
        "config": {
            "project": asdict(cfg.project), # 将dataclass配置对象序列化为字典
            "data": asdict(cfg.data),
            "model": asdict(cfg.model),
            "train": asdict(cfg.train),
            "save": asdict(cfg.save),
        },
    }
    torch.save(artifact, path)
    return path


def load_model_artifact(path: str, device=None):
    """从训练保存的模型文件中恢复模型、标签映射和元数据。"""
    checkpoint = torch.load(path, map_location=device)
    label_map = checkpoint["label_map"]
    # 先重新创建模型结构，再加载权重
    model = BertNERModel(
        pretrain_name=checkpoint["pretrain_name"],
        num_labels=len(label_map),
        dropout=checkpoint.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, label_map, checkpoint
