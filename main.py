import argparse
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from src.config import ProjectConfigLoader
from src.train import train
from src.utils import set_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="实验JSON配置文件路径")
    args = parser.parse_args()

    # 加载配置
    cfg = ProjectConfigLoader(args.config_path)

    # 固定随机种子
    set_seed(cfg.train.seed)

    # 启动训练
    train(cfg=cfg)


