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
            current_type = label[2:] # "B‑LOC"从第2位截取，拿到LOC
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
                entities.add((current_type, start, idx-1))
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
    :param true_labels_list: 所有句子的真实标签列表
    :param pred_labels_list: 所有句子的预测标签列表
    :param id2label: id转标签名
    :return: precision, recall, f1
    '''

    # 全局累加变量，保存整个验证集的总数
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for true_ids, pred_ids in zip(true_labels_list, pred_labels_list):
        # 过滤掉-100，只保留真实汉字
        true_valid = [l for l in true_ids if l != -100]
        pred_valid = [l for l in pred_ids if l != -100]

        # 分别解析真实实体集合、预测实体集合
        true_entities = extract_entities(true_valid, id2label)
        pred_entities = extract_entities(pred_valid, id2label)

        # 交集：真实和预测同时存在的实体 = 预测正确 TP
        tp = len(true_entities & pred_entities)
        # 预测有、真实没有：多识别出来的错误实体 FP
        fp = len(pred_entities - true_entities)
        # 真实有、预测没有：漏掉的实体 FN
        fn = len(true_entities - pred_entities)

        # 累加到全局统计值
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # 防除0保护：分母等于0时，指标直接赋值0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def evaluate(model,data_loader,device,id2label:dict):
    '''
    完整验证/测试评估流程
    :param model: 训练好的BertNERModel
    :param data_loader: 验证集或者测试集DataLoader
    :param device: cuda
    :param id2label:
    :return: 字典包含平均loss、precision、recall、f1
    '''
    model.eval()
    # 累计整个验证集loss总和
    total_loss = 0.0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播，计算loss和logits
            ouputs = model(input_ids,attention_mask=attention_mask,labels=labels)
            total_loss += ouputs["loss"].item()

            preds = torch.argmax(ouputs["logits"], dim=-1)

            # 使用extend展平batch
            batch_pred = preds.cpu().numpy().tolist()
            batch_true = labels.cpu().numpy().tolist()
            all_pred.extend(batch_pred)
            all_true.extend(batch_true)

        # 计算整个验证集平均loss
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
        # 传入全部预测结果，计算实体级P/R/F1
        metrics = calculate_entity_metrics(all_true, all_pred, id2label)
        # 把loss一并加入指标字典返回
        metrics["loss"] = avg_loss

        return metrics
















