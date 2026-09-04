# Demo 2：BERT 中文命名实体识别
## 任务详细分析
### 1.1 任务定义
命名实体识别（Named Entity Recognition, NER）属于**Token级序列标注任务**，给定一个句子，模型需要识别出句子中每个字对应的实体标签，同时判断实体的边界和类型。
- 标注体系：BIO三标签体系
  - `B‑XXX`：实体的起始字（Begin）
  - `I‑XXX`：实体的内部字（Inside）
  - `O`：非实体字（Other）
- 任务输出：句子中每个汉字对应一个标签，输出维度为 `[batch_size, seq_len, num_labels]`
- 与 Demo1 文本分类的本质区别：文本分类是**句子级**（一整句一个标签），NER 是**字级**（每个字一个标签）。

### 1.2 双数据集解析
#### MSRA数据集
- 实体类型（3大类）：PER（人名）、LOC（地名）、ORG（组织机构名）
- 标签总数：`3类实体 × 2(B/I) + 1(O) = 7` 个标签
- 数据划分：训练集5000句，验证集1000句，测试集1000句
- 存储文件：`train_5k.txt`、`dev_1k.txt`、`test_1k.txt`
- 数据格式：每行格式为 `字 标签`，空行分隔不同句子

**MSRA标签映射表**

| 标签编号 | 标签值 | 含义           |
|:--------:|:------:|:--------------:|
| 0        | B‑LOC  | 地名起始字     |
| 1        | B‑ORG  | 机构名起始字   |
| 2        | B‑PER  | 人名起始字     |
| 3        | I‑LOC  | 地名内部字     |
| 4        | I‑ORG  | 机构名内部字   |
| 5        | I‑PER  | 人名内部字     |
| 6        | O      | 非实体汉字     |

#### Weibo微博数据集
- 实体类型（8大类）：`PER.NAM / PER.NOM`、`LOC.NAM / LOC.NOM`、`ORG.NAM / ORG.NOM`、`GPE.NAM / GPE.NOM`
  - NAM：明确命名实体；NOM：泛指类实体
- 标签总数：`8类实体 × 2(B/I) + 1(O) = 17` 个标签
- 数据划分：训练集1350句，验证集269句，测试集270句
- 存储文件：`train.txt`、`dev.txt`、`test.txt`、`class.txt`

**Weibo标签映射表**

| 标签编号 | 标签值      | 含义               |
|:--------:|:-----------:|:------------------:|
| 0        | B‑PER.NOM   | 泛指人名起始       |
| 1        | I‑PER.NOM   | 泛指人名内部       |
| 2        | B‑LOC.NAM   | 专有地名起始       |
| 3        | I‑LOC.NAM   | 专有地名内部       |
| 4        | B‑PER.NAM   | 专有姓名起始       |
| 5        | I‑PER.NAM   | 专有姓名内部       |
| 6        | B‑GPE.NAM   | 专有行政区起始     |
| 7        | I‑GPE.NAM   | 专有行政区内部     |
| 8        | B‑ORG.NAM   | 专有组织起始       |
| 9        | I‑ORG.NAM   | 专有组织内部       |
| 10       | B‑ORG.NOM   | 泛指组织起始       |
| 11       | I‑ORG.NOM   | 泛指组织内部       |
| 12       | B‑LOC.NOM   | 泛指地点起始       |
| 13       | I‑LOC.NOM   | 泛指地点内部       |
| 14       | B‑GPE.NOM   | 泛指行政区起始     |
| 15       | I‑GPE.NOM   | 泛指行政区内部     |
| 16       | O           | 非实体汉字         |

## 2. 项目结构与运行方式
### 2.1 项目结构
```text
demo2/
├── main.py                 # 程序入口，解析config_path后开始训练
├── config.py               # 把JSON配置读取为Python配置对象
├── dataset.py              # 数据读取、Dataset、BERT token标签对齐
├── model.py                # BERT+全连接层组成的序列标注模型
├── train.py                # 训练、验证、保存最优模型、测试
├── evaluate.py             # 验证/测试评估流程
├── metrics.py              # 实体级NER指标，update/compute/reset风格
├── utils.py                # 随机种子、目录创建等工具
├── configs/                # 每个实验的JSON配置文件
└── data/                   # 数据集目录（.gitignore忽略）
```

### 2.2 运行命令
安装依赖：
```bash
pip install -r requirements.txt
```
运行MSRA数据集实验：
```bash
python main.py --config_path configs/msra_bert_wwm.json
python main.py --config_path configs/msra_bert_base.json
```
运行Weibo数据集实验：
```bash
python main.py --config_path configs/weibo_bert_base.json
python main.py --config_path configs/weibo_bert_wwm.json
```

## 3. 实验参数设置
本实验采用统一超参，仅更换预训练模型与数据集，保证变量唯一，实现对照实验。

| 参数项                 | 参数取值说明                                   |
|:---------------------- |:-----------------------------------------|
| 预训练模型              | `bert‑base‑chinese` / `chinese‑bert‑wwm` |
| 优化器                  | AdamW                                    |
| 学习率                  | 2e‑5                                     |
| Batch Size              | 8                                        |
| 最大句子长度            | 128                                      |
| Warmup比例              | 0.1                                      |
| 损失函数                | CrossEntropyLoss（忽略padding位置）            |
| 早停策略                | 连续多轮验证集F1不再提升则停止训练                       |
| 模型保存规则            | 保存验证集F1最优权重，**不用最后一轮epoch权重做测试评估**       |
| 评估方式                | **实体粒度P/R/F1**（边界+类型完全匹配才算正确）            |
| 可视化工具              | SwanLab记录loss、验证集指标变化曲线                  |

## 4. 实验结果对比
### 4.1 四组实验测试集结果汇总
| 数据集 | 预训练模型 | Precision | Recall | Test‑F1 |
|:------:|:----------:|:---------:|:------:|:-------:|
| MSRA | bert‑base‑chinese | 0.9262 | 0.9151 | 0.9206 |
| MSRA | chinese‑bert‑wwm  | 0.9364 | 0.9104 | **0.9232** |
| Weibo | bert‑base‑chinese | 0.6883 | 0.6765 | 0.6823 |
| Weibo | chinese‑bert‑wwm  | 0.6872 | 0.6838 | **0.6855** |

### 4.2 训练过程关键现象
1. MSRA（规范新闻文本）：训练loss快速下降，验证集F1最高可达0.93+，早停生效防止过拟合；
2. Weibo（社交口语文本）：训练数据量更少、标签类别更多，整体性能显著低于MSRA，模型提升空间受限；
3. `chinese‑bert‑wwm` 在两个数据集上F1均微弱优于普通bert‑base，验证**全词掩码预训练对中文实体识别具备增益**；
4. 训练后期train loss无限趋近于0，但验证集F1不再上涨，说明模型出现过拟合，早停策略有效规避该问题。

## 5. 实验分析与总结
四组对照实验中综合性能最优方案：**MSRA数据集 + chinese‑bert‑wwm，测试集F1=0.9232**。

SwanLab实验可视化地址：
https://swanlab.cn/@lr0513/bert_ner_demo2
