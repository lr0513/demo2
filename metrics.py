def extract_entities(label_sequence: list, id2label: dict):
    """
    从标签 id 序列提取完整实体。

    :param label_sequence: 已过滤-100、只保留真实token对应标签的一维id列表
    :param id2label: id转标签名
    :return: 实体集合 {(实体类型, start索引, end索引)}
    """
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
            # 忽略没有B-开头或实体类型不一致的非法I-序列。
            if current_type is None or current_type != label[2:]:
                current_type = None
                start = None
        else:
            if current_type is not None:
                entities.add((current_type, start, idx - 1))
            current_type = None
            start = None

    # 实体一直延伸到句子末尾时，没有后续O触发保存，需要在这里补上。
    if current_type is not None:
        entities.add((current_type, start, len(label_sequence) - 1))

    return entities


class NEREntityMetric:
    """实体级P/R/F1指标，采用Lightning Metric的update/compute风格。"""
    def __init__(self, id2label: dict):
        self.id2label = id2label
        self.reset()

    def reset(self):
        """重置TP/FP/FN，避免上一轮验证的数据污染本次指标"""
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0

    def update(self, true_labels_list: list, pred_labels_list: list):
        """增量累计一批句子的实体级TP/FP/FN。
        :param true_labels_list: 每项是一条句子的真实标签id列表，-100表示非真实token
        :param pred_labels_list: 每项是与真实标签逐token对齐的预测标签id列表
        """
        for true_ids, pred_ids in zip(true_labels_list, pred_labels_list):
            # 成对过滤，避免真实标签去掉[CLS]/[SEP]/PAD后，预测序列仍保留它们造成错位。
            valid_pairs = [(t, p) for t, p in zip(true_ids, pred_ids) if t != -100]
            if not valid_pairs:
                continue

            true_valid = [t for t, _ in valid_pairs]
            pred_valid = [p for _, p in valid_pairs]

            true_entities = extract_entities(true_valid, self.id2label) # 有效汉字对应的真实标签id
            pred_entities = extract_entities(pred_valid, self.id2label) # 有效汉字对应的预测标签id

            self.total_tp += len(true_entities & pred_entities)
            self.total_fp += len(pred_entities - true_entities)
            self.total_fn += len(true_entities - pred_entities)

    def compute(self) -> dict:
        """根据当前累计状态计算precision、recall、f1。"""
        total_positive = self.total_tp + self.total_fp
        total_true = self.total_tp + self.total_fn

        precision = self.total_tp / total_positive if total_positive > 0 else 0.0
        recall = self.total_tp / total_true if total_true > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
