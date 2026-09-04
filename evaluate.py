import torch

def extract_entities(label_squence: list, id2label: dict):
    '''
    从标签id序列提取完整实体
    :param label_squence: 已经过滤掉-100，只有真实汉字对应的标签id，一维列表
    :param id2label: 数字id转回字符串标签（0→B‑PER这种映射）
    :return: 实体集合 {(实体类型, start索引, end索引)}
    '''
    entities = set()
    current_type = None  # 记录当前的实体类型
    start = None  # 当前实体的起始下标

    for idx, label in enumerate(label_squence):
        # 数字id转为可读字符串标签
        label = id2label[label]

        # 情况1：当前标签是B-XXX
        if label.startswith('B-'):
            # 如果遍历到B的时候，前面的实体还没结束，把前面的实体存起来
            if current_type is not None:
                entities.add((current_type, start, idx - 1))
            # 更新为新实体
            current_type = label[2:]  # "B‑LOC"从第2位截取，拿到LOC
            start = idx

        # 情况2：当前标签是I-XXX
        elif label.startswith('I-'):
            # 判断两种非法情况：前面没有B，直接有的I或者I的类型和当前的类型不一致
            if current_type is None or current_type != label[2:]:
                current_type = None
                start = None

        # 情况3：当前标签是O
        else:
            if current_type is not None:
                entities.add((current_type, start, idx - 1))
                # 清空状态，等待下一个B
                current_type = None
                start = None

    # 特殊边界：句子最后几个字符刚好是实体，后面没有O来触发保存，需要单独判断保存
    if current_type is not None:
        entities.add((current_type, start, len(label_squence) - 1))
    return entities


def calculate_entity_metrics(true_labels_list: list, pred_labels_list: list, id2label: dict):
    '''
    计算实体级精确率、召回率、F1
    :param true_labels_list: 每个元素是**单条句子的标签列表**
    :param pred_labels_list: 每个元素是**单条句子的预测标签列表**
    :param id2label: id转标签名
    :return: precision, recall, f1
    '''
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for true_ids, pred_ids in zip(true_labels_list, pred_labels_list):
        # 成对过滤：真值和预测值绑定在一起过滤
        valid_pairs = [(t, p) for t, p in zip(true_ids, pred_ids) if t != -100]
        if valid_pairs:
            true_valid, pred_valid = zip(*valid_pairs)
        else:
            true_valid, pred_valid = [], []

        true_entities = extract_entities(true_valid, id2label)
        pred_entities = extract_entities(pred_valid, id2label)

        tp = len(true_entities & pred_entities)
        fp = len(pred_entities - true_entities)
        fn = len(true_entities - pred_entities)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def evaluate(model, data_loader, device, id2label: dict):
    '''
    完整验证/测试评估流程
    :param model: 训练好的BertNERModel
    :param data_loader: 验证集或者测试集DataLoader
    :param device: cuda
    :param id2label:
    :return: 字典包含平均loss、precision、recall、f1
    '''
    model.eval()
    total_loss = 0.0
    all_true = []   # 存放【每条句子独立的标签列表】
    all_pred = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs["loss"].item()

            preds = torch.argmax(outputs["logits"], dim=-1)

            batch_pred = preds.cpu().numpy().tolist()
            batch_true = labels.cpu().numpy().tolist()

            for t, p in zip(batch_true, batch_pred):
                all_true.append(t)
                all_pred.append(p)

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
    metrics = calculate_entity_metrics(all_true, all_pred, id2label)
    metrics["loss"] = avg_loss
    return metrics
