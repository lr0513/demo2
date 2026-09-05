import argparse
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.config import ProjectConfigLoader
from src.dataset import NERDataset, get_collate_fn, load_ner_data
from src.evaluate import evaluate
from src.model import load_model_artifact
from src.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="加载训练好的模型，单独评估测试集")
    parser.add_argument("--config_path", type=str, required=True, help="实验JSON配置文件路径")
    parser.add_argument("--ckpt_path", type=str, default=None, help="模型文件路径，默认读取最优模型")
    args = parser.parse_args()

    cfg = ProjectConfigLoader(args.config_path)
    set_seed(cfg.train.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.ckpt_path or os.path.join(cfg.save.model_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到模型文件: {ckpt_path}")

    model, label_map, metadata = load_model_artifact(ckpt_path, device=device)
    model.to(device)
    id2label = {v: k for k, v in label_map.items()}

    max_len = int(metadata.get("max_len", cfg.train.max_len))
    tokenizer = AutoTokenizer.from_pretrained(metadata["pretrain_name"])

    test_sentences, test_labels, _ = load_ner_data(
        cfg.data.test_path,
        label_map=label_map,
    )
    test_dataset = NERDataset(test_sentences, test_labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=get_collate_fn(tokenizer, max_len=max_len),
    )

    print(f"使用设备: {device}")
    print(f"测试集句子数: {len(test_dataset)}")
    metrics = evaluate(model, test_loader, device, id2label)
    print("==========测试集评估==========")
    print(f"test precision:{metrics['precision']:.4f} "
          f"recall:{metrics['recall']:.4f} f1:{metrics['f1']:.4f}")
    print(f"test loss:{metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
