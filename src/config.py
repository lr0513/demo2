import json
from dataclasses import dataclass

@dataclass
class ProjectConfig:
    project: str
    experiment_name: str

@dataclass
class ModelConfig:
    pretrain_name: str

@dataclass
class DataConfig:
    train_path: str
    dev_path: str
    test_path: str
    class_path: str = None  # weibo数据集有class.txt，msra为None


@dataclass
class TrainConfig:
    seed: int
    epoch: int
    batch_size: int
    max_len: int
    lr: float
    warmup_rate : float
    early_stop_patience: int
    dropout: float
    grad_clip_norm: float

@dataclass
class SaveConfig:
    model_dir: str
    log_dir: str

class ProjectConfigLoader:
    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.data = DataConfig(**config["data"])
        self.train = TrainConfig(**config["train"])
        self.model = ModelConfig(**config["model"])
        self.save = SaveConfig(**config["save"])
        self.project = ProjectConfig(**config["project"])
