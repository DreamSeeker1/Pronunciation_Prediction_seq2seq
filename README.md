## 项目介绍
利用基本的Encoder-Decoder模型实现seq2seq进行发音预测。代码基于[Nelson Zhao 知乎专栏《机器不学习》的源代码](https://github.com/NELSONZHAO/zhihu/tree/master/basic_seq2seq)修改完成。



## 代码框架
```
TensorFlow 1.3
Python 2.7
Scrapy 1.4
```

## 发音训练数据集
采用 CMUdict version 0.7a，位于根目录。

## 功能介绍
---
### 1. seq2seq模型
---
#### 模型简介
本项目采用了最基本的encoder-decoder模型实现seq2seq功能，支持LSTM/GRU选择，decoder部分使用beam search支持多候选，并且输出各个候选序列概率的对数，没有使用输入逆序以及attention机制。代码主要利用了TensorFlow中[tf.contrib.seq2seq](https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/seq2seq)模块。
LSTM/GRU 作为 RNN cell 可以防止梯度消失。
使用Gradient Clipping 防止梯度爆炸。

原理参考： [从Encoder到Decoder实现Seq2Seq模型](https://zhuanlan.zhihu.com/p/27608348)
![](https://pic4.zhimg.com/v2-278b5920ac2b4fc8c2319c90eaa7f9db_r.png)

##### 基本流程
1. 首先将输入输出序列中的各个元素（单词以字母为单位，发音以音节为单位）拆分，分别加入`<PAD>`，`<UNK>`，`<GO>`，`<EOS>`并且分配id
2. 对输入输出序列进行预处理后（加入`<PAD>`，`<UNK>`，`<GO>`，`<EOS>`）转化为id序列
3. 对id序列进行embedding，将整型向量变为浮点型向量
4. 将输入序列按时间顺序输入Encoder部分
5.  Encoder部分输出接入Decoder部分
    * 训练时Decoder网络每个时间点输入对应的正确结果进行训练
    * 预测或验证时 t - 1 时刻预测结果作为 t 时刻的输入
6. 计算训练结果交叉熵
7. 计算梯度，并且进行 gradient clipping， 反向传播
8. 重复步骤 4 - 7

#### 涉及到的类和函数
* [tf.contrib.rnn.GRUCell](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/GRUCell) RNN 中使用的GRU Cell。
* [tf.contrib.rnn.LSTMCell](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMCell) RNN 中使用的LSTM Cell。
* [tf.contrib.rnn.MultiRNNCell](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell) 将多个RNN Cell按顺序连接起来，方便构成多层RNN。
* [tf.contrib.layers.embed_sequence](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/embed_sequence) 输入符号序列，符号数量和embedding的维度，将符号序列映射成为embedding序列。
* [tf.nn.dynamic_rnn](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn) 根据给的RNNCell构建RNN网络，并且根据输入的序列长度动态计算输出，返回值`output`为每个时刻 t 网络的输出，`state`为网络最终的最终状态。
* [tf.contrib.seq2seq.dynamic_decode](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_decode) 根据采用的decoder进行解码。
* [tf.contrib.seq2seq.BasicDecoder](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BasicDecoder) 一种用于`tf.contrib.seq2seq.dynamic_decode`解码的decoder类型，根据`helper`的不同可以用来训练模型([tf.contrib.seq2seq.TrainingHelper](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/TrainingHelper))和使用模型进行预测([tf.contrib.seq2seq.GreedyEmbeddingHelper](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/GreedyEmbeddingHelper))，常见用法见[TensorFlow参考页面](https://www.tensorflow.org/versions/master/api_guides/python/contrib.seq2seq#Dynamic_Decoding)。
* [tf.contrib.seq2seq.BeamSearchDecoder](https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/seq2seq/BasicDecoder) 采用beam search的方式查找结果，用于模型的infer阶段，根据设定的`beam_width`产生对应数量的候选并给出`score`（概率的对数）。
* [tf.train.AdamOptimizer](https://www.tensorflow.org/versions/master/api_docs/python/tf/train/AdamOptimizer) Adam优化算法，可以采用其他的[算法](https://www.tensorflow.org/versions/master/api_guides/python/train#Optimizers)。
* [tf.clip_by_value](https://www.tensorflow.org/versions/master/api_docs/python/tf/clip_by_value) 根据给定范围对梯度进行剪裁。
##### Beam Search 算法介绍
一种广度优先算法，可以产生多个候选序列。

![](https://raw.githubusercontent.com/yanwii/seq2seq/master/img/beamsearch.png)

[Wiki 介绍](https://en.wikipedia.org/wiki/Beam_search)
[知乎 介绍](https://www.zhihu.com/question/54356960)

*TensorFlow 1.3 中使用[tf.contrib.seq2seq.BeamSearchDecoder](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BeamSearchDecoder)时需要注意[tf.contrib.seq2seq.dynamic_decode](tf.contrib.seq2seq.dynamic_decode)中的`impute_finished`选项必须设置为`FALSE`，否则编译无法通过，参考[issue11598](https://github.com/tensorflow/tensorflow/issues/11598)。*


#### 使用方法
`./tensor_seq`文件夹中为seq2seq模型主体。

**首次运行模型时需要先运行**
```bash
cd ./tensor_seq
python converter.py
python data.py
```
这将在`./tensor_seq/`目录下生成`source_list`,`target_list`与`data.pickle`三个文件。
`converter.py`将数据集中的非单词内容去除，将单词变为小写以后把单词与对应的读音分别存入`source_list`与`target_list`之中。
`data.py`对数据进行进一步的预处理，建立发音，字母和特殊字符与整数之间的映射词典，存入`data.pickle`中方便读取。

与模型相关的参数位于`./tensor_seq/model.py`代码前部。
```python
# Number of Epochs
epochs = 20
# Batch Size
batch_size = 512
# RNN Size
rnn_size = 256
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 40
decoding_embedding_size = encoding_embedding_size
# Learning Rate
learning_rate = 0.001
# cell type 0 for lstm, 1 for GRU
Cell_type = 1
# decoder type 0 for basic, 1 for beam search
Decoder_type = 1
# beam width for beam search decoder
beam_width = 3
# 1 for training, 0 for test the already trained model
isTrain = 1
# display step for training
display_step = 50
```
*调整`batch_size`后注意根据情况调整训练数据集和测试数据集大小*
```python
    train_source = source_int_shuffle[50 * batch_size:]
    train_target = target_int_shuffle[50 * batch_size:]
    valid_source = source_int_shuffle[:50 * batch_size]
    valid_target = target_int_shuffle[:50 * batch_size]
```
设定完成后可以通过运行以下命令对模型进行训练。
```bash
python model.py
```
模型训练过程中会在`./tensor_seq/`目录下生成`graph/`与`model/`两个文件夹，其中`model/`中保存了模型训练过程中的各个状态，用于训练完成后读取模型进行发音预测。`graph/`中保存了计算图的相关信息，以及training loss和validation loss，利用tensorboard工具可以对模型以及相关参数进行可视化。

输入以下命令来开启tensorboard
```bash
tensorboard --logdir ./graph/
```
开启以后只要在浏览器中输入`http://127.0.0.1:6006`即可访问tensorboard。

模型训练完成后将参数`isTrain`的值设置为0后再次运行`model.py`可以对训练好的模型进行测试。
使用方法：
```bash
python model.py
```

### 2. 爬虫介绍
---
位于`./webcrawler`文件夹内，使用[`Scrapy`](https://scrapy.org/)框架。爬虫根据词表爬取其在[iCIBA](http://www.iciba.com/)网站的英式和美式发音并存取，设置了自动切换`user agent`并且限制了爬取速率防止被ban。`result.csv`是根据`cmudict.0.7a`中单词爬取的结果。

#### 使用方法
*运行爬虫前首先在根目录建立词表，每个单词以回车分隔。命名为`word_list`*

词表建立完成后进入爬虫目录，并运行爬虫
```bash
cd ./webcrawler
scrapy crawl ciba -o ./crawl_result.csv
```
这将在当前目录下生成`crawl_result.csv`文件，其中保存了爬取结果。结果分为三列，分别为单词，美音，英音，值为`NULL`表示没有爬取到相应结果，第一行为表头。

### 3. 数据集分割工具
---
位于`./Split_Dataset`文件夹内的`sp.py`将`cmudict.0.7a`中的单词及读音提取出来，去除标点符号的读法，并且随机抽取10000条数据作为测试数据(`testing`)，10000条作为验证数据(`validation`)，剩下的作为训练数据(`training`)。

#### 使用方法
```bash
cd  ./Split_Dataset
python sp.py
```
运行完成后将会在`Split_Dataset`中生成`testing`,`validation`,`training`三个文件。
### TODO
---
+ [ ] 加入attention机制
+ [ ] 采用输入逆序进行训练
