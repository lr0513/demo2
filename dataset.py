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
            label_map = {label: idx for idx, label in enumerate(bio_label_list)}
        else:
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
    '''
    给每个原始字找到它在BERT token序列中的位置，再把该字的标签放到对应位置，特殊token和非首个sub-word统一用-100忽略
    :param tokenizer:
    :param max_len:
    :return:
    '''
    def collate_fn(batch):
        texts = [item[0] for item in batch]
        batch_labels = [item[1] for item in batch]

        encode_result = tokenizer(
            texts,
            padding="max_length", # 保证batch内形状一致
            truncation=True,
            return_tensors="pt",
            max_length=max_len,
            is_split_into_words=True # 告知分词器输入已经是拆分完成的字符序列，不再重新切分文本
        )

        input_ids = encode_result["input_ids"]
        attention_mask = encode_result["attention_mask"]
        batch_size, seq_len = input_ids.shape

        label_tensor = torch.full(
            (batch_size, seq_len),
            fill_value=-100,
            dtype=torch.long
        )

        for i in range(batch_size):
            raw_label = batch_labels[i]
            word_id_list = encode_result.word_ids(batch_index=i) # 第i条样本中，每一个BERT token对应原始输入列表的下标
            prev_word_idx = None # 记录上一个token属于原始句子的哪个词，用于处理“一个原始词被tokenizer切成了多个token”的情况
            for pos, word_idx in enumerate(word_id_list):
                if word_idx is None:
                    # CLS / SEP / PAD
                    label_tensor[i][pos] = -100
                else:
                    # 只给该单词第一个token赋值真实标签，sub‑word保持‑100
                    if word_idx != prev_word_idx:
                        label_tensor[i][pos] = raw_label[word_idx]
                    prev_word_idx = word_idx

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_tensor
        }

    return collate_fn
