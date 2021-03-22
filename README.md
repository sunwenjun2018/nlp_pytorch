# NLP 项目 的  pytorch 实现

## 一、介绍

该项目主要 使用 pytorch 框架 复现 NLP 基础任务上的应用。

## 二、环境

- python 3.7  
- pytorch 1.1  
- tqdm  
- sklearn  
- tensorboardX
  
## 三、项目架构

### 3.1 中文文本分类 任务 textClassifier

#### 3.1.1 数据集

我从[THUCNews](http://thuctc.thunlp.org/)中抽取了20万条新闻标题，已上传至github，文本长度在20到30之间。一共10个类别，每类2万条。

类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。

数据集划分：

数据集|数据量
--|--
训练集|18万
验证集|1万
测试集|1万


#### 3.1.2  [深度学习模型](textClassifier/DL/)

##### 3.1.2.1 模型

- TextCNN
- DPCNN
- FastText
- TextRNN
- TextRCNN
- TextRNN_Att
- Transformer

##### 3.1.2.2 模型效果对比

模型|acc|备注
--|--|--
TextCNN|91.22%|Kim 2014 经典的CNN文本分类
TextRNN|91.12%|BiLSTM 
TextRNN_Att|90.90%|BiLSTM+Attention
TextRCNN|91.54%|BiLSTM+池化
FastText|92.23%|bow+bigram+trigram， 效果出奇的好
DPCNN|91.25%|深层金字塔CNN
Transformer|89.91%|效果较差

##### 3.1.2.3  使用说明

```s
    # 训练并测试：
    # TextCNN
    python run.py --model TextCNN

    # TextRNN
    python run.py --model TextRNN

    # TextRNN_Att
    python run.py --model TextRNN_Att

    # TextRCNN
    python run.py --model TextRCNN

    # FastText, embedding层是随机初始化的
    python run.py --model FastText --embedding random 

    # DPCNN
    python run.py --model DPCNN

    # Transformer
    python run.py --model Transformer
```

#### 3.1.3 [中文文本分类任务 Bert](textClassifier/bert_task/classifier_task/)

##### 3.1.3.1 模型

- Bert-CRF

### 3.2 中文命名实体识别 任务 NER

#### 3.2.1 [BiLSTM-CRF](NER/bilstm_crf_pytorch/)

##### 3.2.1.1 数据集格式

```s
    {"text": "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，", "label": {"name": {"叶老桂": [[9, 11]]}, "company": {"浙商银行": [[0, 3]]}}}
    ...
```

##### 3.2.1.2 运行效果

```s
    12/27/2020 15:04:08 - INFO - root -   Loading features from cached file dataset\cluener\cached_crf-train_bilstm_crf_ner
    336 batches created
    Epoch 1/50
    [Training] 334/336 [============================>.] - ETA: 1s  loss: 1.6212    
    12/27/2020 15:09:17 - INFO - root -   Loading features from cached file dataset\cluener\cached_crf-dev_bilstm_crf_ner
    [Training] 336/336 [==============================] 918.7ms/step  loss: 3.8900  
    42 batches created
    [Evaluating] 41/42 [============================>.] - ETA: 0ss
    12/27/2020 15:09:41 - INFO - root -   
    Epoch: 1 -  loss: 5.8060 - eval_loss: 10.4961 - eval_acc: 0.5807 - eval_recall: 0.6263 - eval_f1: 0.6027 
    12/27/2020 15:09:41 - INFO - root -   
    Epoch 1: eval_f1 improved from 0 to 0.602662490211433
    12/27/2020 15:09:41 - INFO - root -   save model to disk.
    12/27/2020 15:09:41 - INFO - root -   Subject: name - Acc: 0.5699 - Recall: 0.6925 - F1: 0.6252
    [Evaluating] 42/42 [==============================] 578.7ms/step 
    Eval Entity Score: 
    12/27/2020 15:09:41 - INFO - root -   Subject: address - Acc: 0.4749 - Recall: 0.3298 - F1: 0.3892
    12/27/2020 15:09:41 - INFO - root -   Subject: movie - Acc: 0.604 - Recall: 0.596 - F1: 0.6
    12/27/2020 15:09:41 - INFO - root -   Subject: position - Acc: 0.6932 - Recall: 0.7252 - F1: 0.7088
    12/27/2020 15:09:41 - INFO - root -   Subject: organization - Acc: 0.6078 - Recall: 0.7221 - F1: 0.66
    12/27/2020 15:09:41 - INFO - root -   Subject: company - Acc: 0.6385 - Recall: 0.6402 - F1: 0.6394
    12/27/2020 15:09:41 - INFO - root -   Subject: scene - Acc: 0.3861 - Recall: 0.5598 - F1: 0.457
    12/27/2020 15:09:41 - INFO - root -   Subject: government - Acc: 0.6726 - Recall: 0.6073 - F1: 0.6383
    12/27/2020 15:09:41 - INFO - root -   Subject: book - Acc: 0.7128 - Recall: 0.4351 - F1: 0.5403
    12/27/2020 15:09:41 - INFO - root -   Subject: game - Acc: 0.5177 - Recall: 0.7932 - F1: 0.6265
    ...
```

#### 3.2.2 [bert_pytorch](NER/bert_pytorch/)

##### 3.2.2.1 数据集格式

```s
    {"text": "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，", "label": {"name": {"叶老桂": [[9, 11]]}, "company": {"浙商银行": [[0, 3]]}}}
    ...
```

##### 3.2.2.2  运行结果

```s
    12/29/2020 20:35:41 - INFO - root -   Creating features from dataset file at CLUEdatasets/cluener/
    12/29/2020 20:35:41 - INFO - processors.ner_seq -   Writing example 0 of 10748
    12/29/2020 20:35:41 - INFO - processors.ner_seq -   *** Example ***
    12/29/2020 20:35:41 - INFO - processors.ner_seq -   guid: train-0
    12/29/2020 20:35:41 - INFO - processors.ner_seq -   tokens: [CLS] 浙 商 银 行 企 业 信 贷 部 叶 老 桂 博 士 则 从 另 一 个 角 度 对 五 道 门 槛 进 行 了 解 读 。 叶 老 桂 认 为 ， 对 目 前 国 内 商 业 银 行 而 言 ， [SEP]
    12/29/2020 20:35:41 - INFO - processors.ner_seq -   input_ids: 101 3851 1555 7213 6121 821 689 928 6587 6956 1383 5439 3424 1300 1894 1156 794 1369 671 702 6235 2428 2190 758 6887 7305 3546 6822 6121 749 6237 6438 511 1383 5439 3424 6371 711 8024 2190 4680 1184 1744 1079 1555 689 7213 6121 5445 6241 8024 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    12/29/2020 20:35:41 - INFO - processors.ner_seq -   input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    12/29/2020 20:35:41 - INFO - processors.ner_seq -   segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    12/29/2020 20:35:41 - INFO - processors.ner_seq -   label_ids: 31 3 13 13 13 31 31 31 31 31 7 17 17 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    ...

    12/29/2020 20:35:42 - INFO - processors.ner_seq -   Writing example 10000 of 10748
    12/29/2020 20:35:42 - INFO - root -   Saving features into cached file CLUEdatasets/cluener/cached_crf-train_chinese_L-12_H-768_A-12_128_cluener
    12/29/2020 20:35:44 - INFO - root -   ***** Running training *****
    12/29/2020 20:35:44 - INFO - root -     Num examples = 10748
    12/29/2020 20:35:44 - INFO - root -     Num Epochs = 3
    12/29/2020 20:35:44 - INFO - root -     Instantaneous batch size per GPU = 8
    12/29/2020 20:35:44 - INFO - root -     Total train batch size (w. parallel, distributed & accumulation) = 8
    12/29/2020 20:35:44 - INFO - root -     Gradient Accumulation steps = 1
    12/29/2020 20:35:44 - INFO - root -     Total optimization steps = 4032
    [Training] 1/1344 [..............................] - ETA: 43:13  loss: 44.7333 
    D:\program\anaconda\lib\site-packages\torch\optim\lr_scheduler.py:136: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
    [Training] 50/1344 [>.............................] - ETA: 51:24  loss: 35.4704 
    12/29/2020 20:37:44 - INFO - root -   Creating features from dataset file at CLUEdatasets/cluener/
    12/29/2020 20:37:44 - INFO - processors.ner_seq -   Writing example 0 of 1343
    12/29/2020 20:37:44 - INFO - processors.ner_seq -   *** Example ***
    12/29/2020 20:37:44 - INFO - processors.ner_seq -   guid: dev-0
    12/29/2020 20:37:44 - INFO - processors.ner_seq -   tokens: [CLS] 彭 小 军 认 为 ， 国 内 银 行 现 在 走 的 是 台 湾 的 发 卡 模 式 ， 先 通 过 跑 马 圈 地 再 在 圈 的 地 里 面 选 择 客 户 ， [SEP]
    12/29/2020 20:37:44 - INFO - processors.ner_seq -   input_ids: 101 2510 2207 1092 6371 711 8024 1744 1079 7213 6121 4385 1762 6624 4638 3221 1378 3968 4638 1355 1305 3563 2466 8024 1044 6858 6814 6651 7716 1750 1765 1086 1762 1750 4638 1765 7027 7481 6848 2885 2145 2787 8024 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    12/29/2020 20:37:44 - INFO - processors.ner_seq -   input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    12/29/2020 20:37:44 - INFO - processors.ner_seq -   segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    12/29/2020 20:37:44 - INFO - processors.ner_seq -   label_ids: 31 7 17 17 31 31 31 31 31 31 31 31 31 31 31 31 1 11 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    ...

    12/29/2020 20:37:45 - INFO - root -   Saving features into cached file CLUEdatasets/cluener/cached_crf-dev_chinese_L-12_H-768_A-12_512_cluener
    
    12/29/2020 20:37:46 - INFO - root -   ***** Running evaluation  *****
    12/29/2020 20:37:46 - INFO - root -     Num examples = 1343
    12/29/2020 20:37:46 - INFO - root -     Batch size = 8
    [Evaluating] 167/168 [============================>.] - ETA: 0ss
    12/29/2020 20:39:21 - INFO - root -   

    12/29/2020 20:39:21 - INFO - root -   ***** Eval results  *****
    12/29/2020 20:39:21 - INFO - root -    acc: 0.0000 - recall: 0.0000 - f1: 0.0000 - loss: 44.2423 
    12/29/2020 20:39:21 - INFO - root -   ***** Entity results  *****
    12/29/2020 20:39:21 - INFO - root -   ******* address results ********
    12/29/2020 20:39:21 - INFO - root -    acc: 0.0000 - recall: 0.0000 - f1: 0.0000 
    12/29/2020 20:39:21 - INFO - root -   ******* book results ********
    12/29/2020 20:39:21 - INFO - root -    acc: 0.0000 - recall: 0.0000 - f1: 0.0000 
    12/29/2020 20:39:21 - INFO - root -   ******* company results ********
    12/29/2020 20:39:21 - INFO - root -    acc: 0.0000 - recall: 0.0000 - f1: 0.0000 
    12/29/2020 20:39:21 - INFO - root -   ******* game results ********
    12/29/2020 20:39:21 - INFO - root -    acc: 0.0000 - recall: 0.0000 - f1: 0.0000 
    12/29/2020 20:39:21 - INFO - root -   ******* government results ********
    12/29/2020 20:39:21 - INFO - root -    acc: 0.0000 - recall: 0.0000 - f1: 0.0000 
    12/29/2020 20:39:21 - INFO - root -   ******* movie results ********
    12/29/2020 20:39:21 - INFO - root -    acc: 0.0000 - recall: 0.0000 - f1: 0.0000 
    12/29/2020 20:39:21 - INFO - root -   ******* name results ********
    12/29/2020 20:39:21 - INFO - root -    acc: 0.0000 - recall: 0.0000 - f1: 0.0000 
    12/29/2020 20:39:21 - INFO - root -   ******* organization results ********
    12/29/2020 20:39:21 - INFO - root -    acc: 0.0000 - recall: 0.0000 - f1: 0.0000 
    12/29/2020 20:39:21 - INFO - root -   ******* position results ********
    12/29/2020 20:39:21 - INFO - root -    acc: 0.0000 - recall: 0.0000 - f1: 0.0000 
    12/29/2020 20:39:21 - INFO - root -   ******* scene results ********
    12/29/2020 20:39:21 - INFO - root -    acc: 0.0000 - recall: 0.0000 - f1: 0.0000 
    12/29/2020 20:39:21 - INFO - models.transformers.configuration_utils -   Configuration saved in outputs/cluener_output/bert\checkpoint-50\config.json
    [Evaluating] 168/168 [==============================] 568.8ms/step
    12/29/2020 20:39:22 - INFO - models.transformers.modeling_utils -   Model weights saved in outputs/cluener_output/bert\checkpoint-50\pytorch_model.bin
    12/29/2020 20:39:22 - INFO - root -   Saving model checkpoint to outputs/cluener_output/bert\checkpoint-50
    D:\program\anaconda\lib\site-packages\torch\optim\lr_scheduler.py:216: UserWarning: Please also save or load the state of the optimizer when saving or loading the scheduler.
    warnings.warn(SAVE_STATE_WARNING, UserWarning)
    12/29/2020 20:39:32 - INFO - root -   Saving optimizer and scheduler states to outputs/cluener_output/bert\checkpoint-50
    [Training] 100/1344 [=>............................] - ETA: 1:13:15  loss: 24.8498 
    12/29/2020 20:41:38 - INFO - root -   Creating features from dataset file at CLUEdatasets/cluener/
    12/29/2020 20:41:38 - INFO - processors.ner_seq -   Writing example 0 of 1343
    12/29/2020 20:41:38 - INFO - processors.ner_seq -   *** Example ***
    12/29/2020 20:41:38 - INFO - processors.ner_seq -   guid: dev-0
    12/29/2020 20:41:38 - INFO - processors.ner_seq -   tokens: [CLS] 彭 小 军 认 为 ， 国 内 银 行 现 在 走 的 是 台 湾 的 发 卡 模 式 ， 先 通 过 跑 马 圈 地 再 在 圈 的 地 里 面 选 择 客 户 ， [SEP]
    12/29/2020 20:41:38 - INFO - processors.ner_seq -   input_ids: 101 2510 2207 1092 6371 711 8024 1744 1079 7213 6121 4385 1762 6624 4638 3221 1378 3968 4638 1355 1305 3563 2466 8024 1044 6858 6814 6651 7716 1750 1765 1086 1762 1750 4638 1765 7027 7481 6848 2885 2145 2787 8024 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    12/29/2020 20:41:38 - INFO - processors.ner_seq -   input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    12/29/2020 20:41:38 - INFO - processors.ner_seq -   segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    12/29/2020 20:41:38 - INFO - processors.ner_seq -   label_ids: 31 7 17 17 31 31 31 31 31 31 31 31 31 31 31 31 1 11 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    ...
    
    12/29/2020 20:41:39 - INFO - root -   Saving features into cached file CLUEdatasets/cluener/cached_crf-dev_chinese_L-12_H-768_A-12_512_cluener
    12/29/2020 20:41:40 - INFO - root -   ***** Running evaluation  *****
    12/29/2020 20:41:40 - INFO - root -     Num examples = 1343
    12/29/2020 20:41:40 - INFO - root -     Batch size = 8
    [Evaluating] 167/168 [============================>.] - ETA: 0ss
    12/29/2020 20:43:24 - INFO - root -   

    12/29/2020 20:43:24 - INFO - root -   ***** Eval results  *****
    12/29/2020 20:43:24 - INFO - root -    acc: 0.8017 - recall: 0.0316 - f1: 0.0608 - loss: 29.8443 
    12/29/2020 20:43:24 - INFO - root -   ***** Entity results  *****
    12/29/2020 20:43:24 - INFO - root -   ******* address results ********
    12/29/2020 20:43:24 - INFO - root -    acc: 0.2000 - recall: 0.0027 - f1: 0.0053 
    12/29/2020 20:43:24 - INFO - root -   ******* book results ********
    12/29/2020 20:43:24 - INFO - root -    acc: 0.0000 - recall: 0.0000 - f1: 0.0000 
    12/29/2020 20:43:24 - INFO - root -   ******* company results ********
    12/29/2020 20:43:24 - INFO - root -    acc: 1.0000 - recall: 0.0079 - f1: 0.0157 
    12/29/2020 20:43:24 - INFO - root -   ******* game results ********
    12/29/2020 20:43:24 - INFO - root -    acc: 0.0000 - recall: 0.0000 - f1: 0.0000 
    12/29/2020 20:43:24 - INFO - root -   ******* government results ********
    12/29/2020 20:43:24 - INFO - root -    acc: 0.0000 - recall: 0.0000 - f1: 0.0000 
    12/29/2020 20:43:24 - INFO - root -   ******* movie results ********
    12/29/2020 20:43:24 - INFO - root -    acc: 0.0000 - recall: 0.0000 - f1: 0.0000 
    12/29/2020 20:43:24 - INFO - root -   ******* name results ********
    12/29/2020 20:43:24 - INFO - root -    acc: 0.8288 - recall: 0.1978 - f1: 0.3194 
    12/29/2020 20:43:24 - INFO - root -   ******* organization results ********
    12/29/2020 20:43:24 - INFO - root -    acc: 1.0000 - recall: 0.0027 - f1: 0.0054 
    12/29/2020 20:43:24 - INFO - root -   ******* position results ********
    12/29/2020 20:43:24 - INFO - root -    acc: 0.0000 - recall: 0.0000 - f1: 0.0000 
    12/29/2020 20:43:24 - INFO - root -   ******* scene results ********
    12/29/2020 20:43:24 - INFO - root -    acc: 0.0000 - recall: 0.0000 - f1: 0.0000 
    12/29/2020 20:43:24 - INFO - models.transformers.configuration_utils -   Configuration saved in outputs/cluener_output/bert\checkpoint-100\config.json
    [Evaluating] 168/168 [==============================] 620.8ms/step
```




