# Demo 2：BERT 中文命名实体识别
## 任务详细分析
### 1.1 任务定义
&nbsp;&nbsp;命名实体识别（Named Entity Recognition, NER）属于**Token级序列标注任务**，给定一个句子，模型需要识别出句子中每个字对应的实体标签，同时判断实体的边界和类型。
 

- 标注体系：BIO三标签体系
  - `B-XXX`：实体的起始字（Begin）
  - `I-XXX`：实体的内部字（Inside）
  - `O`：非实体字（Other）
- 任务输出：句子中每个汉字对应一个标签，输出维度为 `[batch_size, seq_len, num_labels]`
- 与 Demo1 文本分类的本质区别：文本分类是**句子级**（一整句一个标签），NER 是**字级**（每个字一个标签）。

### 1.2 双数据集解析
1. MSRA数据集
  - 实体类型（3大类）：
    - PER：人名
    - LOC：地名
    - ORG：组织机构名
  - 标签总数：`3类实体 × 2(B/I) + 1(O) = 7` 个标签
  - 文件：`train_5k.txt`（训练集）、`dev_1k.txt`（验证集）、`test_1k.txt`（测试集）
  - 格式：每行格式为 `字 标签`，空行分隔两个句子
2. Weibo微博数据集
  - 实体类型（8大类）：
    - `PER.NAM` / `PER.NOM`：命名人名 / 泛指人称
    - `LOC.NAM` / `LOC.NOM`：命名地点 / 泛指地点
    - `ORG.NAM` / `ORG.NOM`：命名组织 / 泛指组织
    - `GPE.NAM` / `GPE.NOM`：命名行政区 / 泛指行政区
  - 标签总数：`8类实体 × 2(B/I) + 1(O) = 17` 个标签
- 文件：`train.txt`、`dev.txt`、`test.txt`、`class.txt`（实体类别清单）








