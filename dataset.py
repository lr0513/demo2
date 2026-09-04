import os

import torch
from torch.utils.data import Dataset


def load_ner_data(file_path: str, label_map: dict = None, class_path: str = None):
    sentences = []
    label_lists = []
    current_sentence = []
    current_label = []

    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 空行代表句子结束
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    label_lists.append(current_label)
                    current_sentence = []
                    current_label = []
                continue
            parts = line.split()
            if len(parts) >= 2:
                char, label = parts[0], parts[1]
                current_sentence.append(char)
                current_label.append(label)

    # 处理最后一句
    if current_sentence:
        sentences.append(current_sentence)
        label_lists.append(current_label)

    # 构建标签映射
    if label_map is None:
        if class_path is not None and os.path.exists(class_path):
            entity_types = []
            with open(class_path, 'r', encoding="utf-8") as f:
                for line in f:
                    t = line.strip()
                    if t:
                        entity_types.append(t)
            # 生成完整BIO标签列表
            bio_label_list = []
            for type in entity_types:
                bio_label_list.append(f"B-{type}")
                bio_label_list.append(f"I-{type}")
            bio_label_list.append("O")
            label_map = {label:idx for idx, label in enumerate(bio_label_list)}
        # MSRA数据集：自动统计标签
        all_labels = set()
        for labels in label_lists:
            all_labels.update(labels)
        all_labels = sorted(list(all_labels))
        label_map = {label: idx for idx, label in enumerate(all_labels)}

    # 标签转id
    label_ids = []
    for labels in label_lists:
        idx = [label_map[label] for label in labels]
        label_ids.append(idx)
    return sentences, label_ids, label_map


class NERDataset(Dataset):
    def __init__(self, sentences, label_ids):
        self.sentences = sentences
        self.label_ids = label_ids

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text = self.sentences[idx]
        label = self.label_ids[idx]
        return text, label


def get_collate_fn(tokenizer, max_len):
    def collate_fn(batch):
        '''
        自定义批次处理函数，DataLoader自动调用
        :param batch: 一个batch的数据，格式 [(sent1,label1),(sent2,label2)...]
        :return: dict: input_ids, attention_mask, labels
        '''
        # 拆分batch：句子列表、标签列表
        texts = [item[0] for item in batch]
        batch_labels = [item[1] for item in batch]
        # 将汉字列表拼成完整字符串，给分词器处理
        text_strings = ["".join(word_list) for word_list in texts]
        encode_result = tokenizer(
            text_strings,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=max_len
        )

        input_ids = encode_result["input_ids"]
        attention_mask = encode_result["attention_mask"]
        batch_size, seq_len = input_ids.shape

        # 初始化标签全部填充 -100，CrossEntropyLoss自动忽略该位置
        label_tensor = torch.full((batch_size, seq_len), fill_value=-100, dtype=torch.long)

        # token与原始汉字标签对齐
        for i in range(batch_size):
            raw_label = batch_labels[i]
            word_id_list = encode_result.word_ids(batch_index=i)
            for pos, word_idx in enumerate(word_id_list):
                if word_idx is not None:
                    label_tensor[i][pos] = raw_label[word_idx]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_tensor
        }

    return collate_fn
