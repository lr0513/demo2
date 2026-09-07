import argparse
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer

from src.config import ProjectConfigLoader
from src.data_module import NERDataModule
from src.evaluate import evaluate
from src.model import BertNERModel
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

    model, label_map, metadata = BertNERModel.load_artifact(ckpt_path, device=device)
    model.to(device)
    id2label = {v: k for k, v in label_map.items()}

    max_len = int(metadata.get("max_len", cfg.train.max_len))
    tokenizer = AutoTokenizer.from_pretrained(metadata["pretrain_name"])

    data_module = NERDataModule(
        cfg,
        tokenizer=tokenizer,
        label_map=label_map,
        max_len=max_len,
    )
    data_module.setup(splits=("test",))
    test_loader = data_module.test_dataloader()

    print(f"使用设备: {device}")
    print(f"测试集句子数: {len(data_module.datasets['test'])}")
    metrics = evaluate(model, test_loader, device, id2label)
    print("==========测试集评估==========")
    print(f"test precision:{metrics['precision']:.4f} "
          f"recall:{metrics['recall']:.4f} f1:{metrics['f1']:.4f}")
    print(f"test loss:{metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
