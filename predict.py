import argparse
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer

from src.config import ProjectConfigLoader
from src.model import load_model_artifact
from src.utils import extract_entities


def predict_text(text, model, tokenizer, id2label, device, max_len):
    chars = list(text.strip().replace(" ", ""))
    encoded = tokenizer(
        chars,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        pred_ids = torch.argmax(outputs["logits"], dim=-1)[0].cpu().tolist()

    # 取batch中第0条样本，建立编码后的token和输入的原始汉字之间的映射关系
    word_ids = encoded.word_ids(0)
    pred_label_ids = []
    prev_word_idx = None
    for pos, word_idx in enumerate(word_ids):
        # 防止sub-word拆分重复预测
        if word_idx is not None and word_idx != prev_word_idx:
            pred_label_ids.append(pred_ids[pos])
        prev_word_idx = word_idx

    print(f"输入句子: {text.strip()}")
    print("逐字预测:")
    for char, label_id in zip(chars, pred_label_ids):
        print(f"  {char} -> {id2label[label_id]}")

    entities = extract_entities(pred_label_ids, id2label)
    print("识别实体:")
    if not entities:
        print("  （未识别到实体）")
    # 按实体起始位置排序输出
    for entity_type, start, end in sorted(entities):
        span = "".join(chars[start:end + 1])
        print(f"  {entity_type}: {span} [位置 {start}-{end}]")


def main():
    parser = argparse.ArgumentParser(description="输入一句话进行 NER 预测")
    parser.add_argument("--config_path", type=str, required=True, help="实验JSON配置文件路径")
    parser.add_argument("--ckpt_path", type=str, default=None, help="模型文件路径，默认读取最优模型")
    parser.add_argument("--text", type=str, default=None, help="要预测的一句话")
    args = parser.parse_args()

    cfg = ProjectConfigLoader(args.config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.ckpt_path or os.path.join(cfg.save.model_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到模型文件: {ckpt_path}")

    model, label_map, metadata = load_model_artifact(ckpt_path, device=device)
    model.to(device)
    id2label = {v: k for k, v in label_map.items()}
    tokenizer = AutoTokenizer.from_pretrained(metadata["pretrain_name"])
    max_len = int(metadata.get("max_len", cfg.train.max_len))

    if args.text:
        predict_text(args.text, model, tokenizer, id2label, device, max_len)
        return

    print("请输入要识别的一句话，输入 exit/quit 结束")
    while True:
        text = input("> ").strip()
        if text.lower() in {"exit", "quit", "q"}:
            break
        if text:
            predict_text(text, model, tokenizer, id2label, device, max_len)
            print()


if __name__ == "__main__":
    main()
