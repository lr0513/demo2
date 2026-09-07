import os

import swanlab
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.dataset import NERDataset
from src.evaluate import evaluate
from src.model import BertNERModel
from src.utils import mkdir_if_not_exist


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    mkdir_if_not_exist(cfg.save.model_dir)
    mkdir_if_not_exist(cfg.save.log_dir)

    swanlab.init(
        project=cfg.project.project,
        experiment_name=cfg.project.experiment_name,
        config={
            "model": cfg.model.__dict__,
            "train": cfg.train.__dict__,
            "data": cfg.data.__dict__,
            "device": str(device),
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrain_name)
    train_dataset = NERDataset.from_file(
        tokenizer=tokenizer,
        file_path=cfg.data.train_path,
        max_len=cfg.train.max_len,
        class_path=cfg.data.class_path,
    )
    label_map = train_dataset.label_map
    dev_dataset = NERDataset.from_file(
        tokenizer=tokenizer,
        file_path=cfg.data.dev_path,
        max_len=cfg.train.max_len,
        label_map=label_map,
    )

    id2label = {v: k for k, v in label_map.items()}
    num_labels = len(label_map)
    print(f"标签数量: {num_labels}")
    print("标签映射:", label_map)

    swanlab.config["label_map"] = label_map

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=dev_dataset.collate_fn,
    )
    print(f"数据集加载完成：训练集{len(train_dataset)}句，验证集{len(dev_dataset)}句")

    model = BertNERModel(
        pretrain_name=cfg.model.pretrain_name,
        num_labels=num_labels,
        dropout=cfg.train.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.train.lr)
    total_steps = len(train_loader) * cfg.train.epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg.train.warmup_rate),
        num_training_steps=total_steps,
    )

    best_f1 = -1.0
    early_stop_count = 0

    for epoch in range(cfg.train.epoch):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            output = model(input_ids, attention_mask, labels)
            loss = output["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"\nEpoch {epoch + 1} train loss: {avg_train_loss:.4f}")

        dev_metrics = evaluate(model, dev_loader, device, id2label)
        dev_precision = dev_metrics["precision"]
        dev_recall = dev_metrics["recall"]
        dev_f1 = dev_metrics["f1"]
        print(f"Epoch {epoch + 1} dev precision:{dev_precision:.4f} recall:{dev_recall:.4f} f1:{dev_f1:.4f}")

        swanlab.log({
            "train/loss": avg_train_loss,
            "dev/precision": dev_precision,
            "dev/recall": dev_recall,
            "dev/f1": dev_f1,
        })

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            early_stop_count = 0
            save_path = os.path.join(cfg.save.model_dir, "best_model.pt")
            model.save_artifact(save_path, label_map, cfg)
            print(f"保存最优模型，best_f1={best_f1:.4f}，路径={save_path}")
        else:
            early_stop_count += 1
            if early_stop_count >= cfg.train.early_stop_patience:
                print("触发早停机制，结束训练")
                break

    print(f"\n训练完成，验证集最优 F1={best_f1:.4f}，最优模型已保存")
    swanlab.finish()
