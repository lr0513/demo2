import torch

from metrics import NEREntityMetric


def evaluate(model, data_loader, device, id2label: dict):
    '''
    完整验证/测试评估流程。

    :param model: 训练好的 BertNERModel
    :param data_loader: 验证集或者测试集 DataLoader
    :param device: cuda
    :param id2label: id 转标签名
    :return: 字典包含平均loss、precision、recall、f1
    '''
    model.eval()
    metric = NEREntityMetric(id2label)
    total_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs["loss"].item()

            preds = torch.argmax(outputs["logits"], dim=-1)

            metric.update(
                true_labels_list=labels.cpu().tolist(),
                pred_labels_list=preds.cpu().tolist(),
            )

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
    metrics = metric.compute()
    metrics["loss"] = avg_loss
    return metrics
