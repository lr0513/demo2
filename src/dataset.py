import os

import torch
from torch.utils.data import Dataset


class NERDataset(Dataset):
    """负责读取NER数据、构建标签映射，并把原始标签对齐到BERT token。"""

    def __init__(self, sentences, label_ids, label_map, tokenizer, max_len):
        self.sentences = sentences
        self.label_ids = label_ids
        self.label_map = label_map
        self.tokenizer = tokenizer
        self.max_len = max_len

    @classmethod
    def from_file(cls, tokenizer, file_path, max_len, label_map=None, class_path=None):
        """从`字 标签`格式文件构建数据集。"""
        sentences, raw_labels = cls._read_ner_file(file_path)
        if label_map is None:
            label_map = cls._build_label_map(class_path, raw_labels)

        label_ids = [
            [label_map[label] for label in labels]
            for labels in raw_labels
        ]
        return cls(sentences, label_ids, label_map, tokenizer, max_len)

    @staticmethod
    def _read_ner_file(file_path):
        sentences = []
        raw_labels = []
        current_sentence = []
        current_labels = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # 空行代表句子结束
                if not line:
                    if current_sentence:
                        sentences.append(current_sentence)
                        raw_labels.append(current_labels)
                        current_sentence = []
                        current_labels = []
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    current_sentence.append(parts[0])
                    current_labels.append(parts[1])

        # 处理最后一句
        if current_sentence:
            sentences.append(current_sentence)
            raw_labels.append(current_labels)

        return sentences, raw_labels

    @staticmethod
    def _build_label_map(class_path, raw_labels):
        '''构建标签映射'''
        if class_path and os.path.exists(class_path):
            entity_types = []
            with open(class_path, "r", encoding="utf-8") as f:
                for line in f:
                    entity_type = line.strip()
                    if entity_type:
                        entity_types.append(entity_type)

            # 生成完整BIO标签
            bio_labels = []
            for entity_type in entity_types:
                bio_labels.append(f"B-{entity_type}")
                bio_labels.append(f"I-{entity_type}")
            bio_labels.append("O")
            return {label: idx for idx, label in enumerate(bio_labels)}

        # MSRA数据集：自动统计标签
        all_labels = sorted({label for labels in raw_labels for label in labels})
        return {label: idx for idx, label in enumerate(all_labels)}

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.label_ids[idx]

    def collate_fn(self, batch):
        """把一个batch编码为BERT输入，同时把标签放到对应token位置。"""
        texts = [item[0] for item in batch]
        batch_labels = [item[1] for item in batch]

        encode_result = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.max_len,
            is_split_into_words=True, # 告知分词器输入已经是拆分完成的字符序列，不再重新切分文本
        )

        input_ids = encode_result["input_ids"]
        attention_mask = encode_result["attention_mask"]
        batch_size, seq_len = input_ids.shape
        label_tensor = torch.full(
            (batch_size, seq_len),
            fill_value=-100,
            dtype=torch.long,
        )

        for batch_idx in range(batch_size):
            raw_label = batch_labels[batch_idx]
            word_id_list = encode_result.word_ids(batch_index=batch_idx) # 第i条样本中，每一个BERT token对应原始输入列表的下标
            prev_word_idx = None # 记录上一个token属于原始句子的哪个词，用于处理“一个原始词被tokenizer切成了多个token”的情况

            for pos, word_idx in enumerate(word_id_list):
                if word_idx is None:
                    # [CLS]、[SEP]、PAD等位置没有真实标签
                    continue

                # 只给该单词第一个token赋值真实标签，sub‑word保持‑100
                if word_idx != prev_word_idx:
                    label_tensor[batch_idx][pos] = raw_label[word_idx]
                prev_word_idx = word_idx

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_tensor,
        }
