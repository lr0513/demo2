import torch
from torch.utils.data import DataLoader

from src.dataset import NERDataset


class NERDataModule:
    """统一管理train/dev/test数据集和DataLoader的Lightning风格数据模块。"""

    REQUIRED_SPLITS = ("train", "dev", "test")

    def __init__(self, cfg, tokenizer, label_map=None, max_len=None, num_workers=0):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_len = max_len or cfg.train.max_len
        self.batch_size = cfg.train.batch_size
        self.num_workers = num_workers
        self.datasets = {}

    @property
    def num_labels(self):
        if self.label_map is None:
            raise RuntimeError("label_map is not ready. Call setup() first.")
        return len(self.label_map)

    @property
    def id2label(self):
        if self.label_map is None:
            raise RuntimeError("label_map is not ready. Call setup() first.")
        return {label_id: label for label, label_id in self.label_map.items()}

    def setup(self, splits=REQUIRED_SPLITS):
        """按需要准备数据，train.py可只准备train/dev，test.py只准备 test。"""
        # 检查传入的划分名称是否合法
        invalid_splits = [split for split in splits if split not in self.REQUIRED_SPLITS]
        if invalid_splits:
            raise ValueError(f"Unknown splits: {invalid_splits}")

        missing_splits = [
            split for split in splits
            if not getattr(self.cfg.data, f"{split}_path", None)
        ]
        if missing_splits:
            raise FileNotFoundError(f"Missing data path for: {missing_splits}")

        self.datasets = {}
        if "train" in splits:
            train_dataset = NERDataset.from_file(
                tokenizer=self.tokenizer,
                file_path=self.cfg.data.train_path,
                max_len=self.max_len,
                label_map=self.label_map,
                class_path=self.cfg.data.class_path,
            )
            self.label_map = self.label_map or train_dataset.label_map
            self.datasets["train"] = train_dataset

        if self.label_map is None:
            raise RuntimeError("Cannot prepare dev/test without a label_map. Include train or pass label_map.")

        for split in splits:
            if split == "train":
                continue
            data_path = getattr(self.cfg.data, f"{split}_path")
            self.datasets[split] = NERDataset.from_file(
                tokenizer=self.tokenizer,
                file_path=data_path,
                max_len=self.max_len,
                label_map=self.label_map,
            )

    def train_dataloader(self):
        return self._dataloader("train", shuffle=True)

    def dev_dataloader(self):
        return self._dataloader("dev", shuffle=False)

    def test_dataloader(self):
        return self._dataloader("test", shuffle=False)

    def _dataloader(self, split, shuffle):
        if split not in self.datasets:
            raise RuntimeError(f"Call setup() with split '{split}' before requesting a data loader.")

        dataset = self.datasets[split]
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_fn,
            pin_memory=torch.cuda.is_available(),
        )
