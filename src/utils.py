import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mkdir_if_not_exist(path: str):
    """文件夹不存在则创建。"""
    if not os.path.exists(path):
        os.makedirs(path)


class NEREntityMetric:
    """实体级 P/R/F1 指标，采用 Lightning Metric 的 update/compute/reset 风格。"""

    @staticmethod
    def extract_entities(label_sequence: list, id2label: dict):
        """从BIO标签id序列还原完整实体。"""
        entities = set()
        current_type = None
        start = None

        for idx, label_id in enumerate(label_sequence):
            label = id2label[label_id]

            if label.startswith("B-"):
                if current_type is not None:
                    entities.add((current_type, start, idx - 1))
                current_type = label[2:]
                start = idx
            elif label.startswith("I-"):
                if current_type is None or current_type != label[2:]:
                    current_type = None
                    start = None
            else:
                if current_type is not None:
                    entities.add((current_type, start, idx - 1))
                current_type = None
                start = None

        if current_type is not None:
            entities.add((current_type, start, len(label_sequence) - 1))

        return entities

    def __init__(self, id2label: dict):
        self.id2label = id2label
        self.reset()

    def reset(self):
        """重置累计状态，避免上一轮评估污染本次结果。"""
        # TP：模型预测是实体，本身的确是实体
        # FP：模型预测是实体，但本身不是
        # FN：本来就是实体，但模型没识别出来
        self.total_tp = 0
        self.total_pred = 0  # TP + FP：模型预测出的全部实体
        self.total_gold = 0  # TP + FN：数据里全部真实标注实体

    def update(self, true_labels_list: list, pred_labels_list: list):
        """增量累计一批句子的TP、预测实体数和真实实体数。"""
        for true_ids, pred_ids in zip(true_labels_list, pred_labels_list):
            valid_pairs = [
                (true_id, pred_id)
                for true_id, pred_id in zip(true_ids, pred_ids)
                if true_id != -100
            ]
            if not valid_pairs:
                continue

            true_valid = [true_id for true_id, _ in valid_pairs]
            pred_valid = [pred_id for _, pred_id in valid_pairs]

            true_entities = self.extract_entities(true_valid, self.id2label)
            pred_entities = self.extract_entities(pred_valid, self.id2label)

            tp = len(true_entities & pred_entities)
            self.total_tp += tp
            self.total_pred += len(pred_valid)
            self.total_gold += len(true_valid)

    def compute(self) -> dict:
        """根据累计状态计算 precision、recall、f1。"""
        # 精确率
        precision = self.total_tp / self.total_pred if self.total_pred > 0 else 0.0
        # 全部真正的实体，模型成功抓到了多少
        recall = self.total_tp / self.total_gold if self.total_gold > 0 else 0.0
        # 精确率和召回率的平衡分数
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
