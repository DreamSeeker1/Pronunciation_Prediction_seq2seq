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
### 1. seq2seq模型
---
#### 模型简介
本项目采用了最基本的encoder-decoder模型实现seq2seq功能，支持LSTM/GRU选择，decoder部分使用beam search支持多候选，并且输出各个候选序列概率的对数，没有使用输入逆序以及attention机制。代码主要利用了TensorFlow中[tf.contrib.seq2seq](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/seq2seq)模块。

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
* [tf.contrib.rnn.GRUCell](https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/ops/rnn_cell_impl.py) RNN 中使用的GRU Cell。
* [tf.contrib.rnn.LSTMCell](https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/ops/rnn_cell_impl.py) RNN 中使用的LSTM Cell。
* [tf.contrib.rnn.MultiRNNCell](https://www.github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/ops/rnn_cell_impl.py) 将多个RNN Cell按顺序连接起来，方便构成多层RNN。
* [tf.contrib.layers.embed_sequence](https://www.github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/contrib/layers/python/layers/encoders.py) 输入符号序列，符号数量和embedding的维度，将符号序列映射成为embedding序列。
* [tf.nn.dynamic_rnn](https://www.github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/ops/rnn.py) 根据给的RNNCell构建RNN网络，并且根据输入的序列长度动态计算输出，返回值`output`为每个时刻 t 网络的输出，`state`为网络最终的最终状态。
* [tf.contrib.seq2seq.dynamic_decode](https://www.github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/contrib/seq2seq/python/ops/decoder.py) 根据采用的decoder进行解码。
* [tf.contrib.seq2seq.BasicDecoder](https://www.github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/contrib/seq2seq/python/ops/basic_decoder.py) 一种用于`tf.contrib.seq2seq.dynamic_decode`解码的decoder类型，根据`helper`的不同可以用来训练模型([tf.contrib.seq2seq.TrainingHelper](https://www.github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/contrib/seq2seq/python/ops/helper.py))和使用模型进行预测([tf.contrib.seq2seq.GreedyEmbeddingHelper](https://www.github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/contrib/seq2seq/python/ops/helper.py))，常见用法见如下。


```python

cell = # instance of RNNCell

if mode == "train":
  helper = tf.contrib.seq2seq.TrainingHelper(
    input=input_vectors,
    sequence_length=input_lengths)
elif mode == "infer":
  helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
      embedding=embedding,
      start_tokens=tf.tile([GO_SYMBOL], [batch_size]),
      end_token=END_SYMBOL)

decoder = tf.contrib.seq2seq.BasicDecoder(
    cell=cell,
    helper=helper,
    initial_state=cell.zero_state(batch_size, tf.float32))
outputs, _ = tf.contrib.seq2seq.dynamic_decode(
   decoder=decoder,
   output_time_major=False,
   impute_finished=True,
   maximum_iterations=20)
   
```


* [tf.contrib.seq2seq.BeamSearchDecoder](https://www.github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/seq2seq/python/ops/beam_search_decoder.py) 采用beam search的方式查找结果，用于模型的infer阶段，根据设定的`beam_width`产生对应数量的候选并给出`score`（概率的对数）。
* [tf.train.AdamOptimizer](https://www.github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/adam.py) Adam优化算法。
* [tf.train.RMSPropOptimizer](https://www.github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/training/rmsprop.py)RMSProp优化算法。
* [tf.train.GradientDescentOptimizer](https://www.github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/training/gradient_descent.py)梯度下降优化算法。也可以采用TensorFlow中提供的其他[算法](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/training)。
* [tf.clip_by_norm](https://www.github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/clip_ops.py) 根据给定范数范围对梯度进行剪裁。
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
./initialize.sh
```
这将在`./tensor_seq/dataset`目录下生成`source_list_training`,`target_list_training`与`data.pickle`等文件。
`converter.py`将数据集之中的单词与读音分别存入对应的`source_list`与`target_list`之中。
`data.py`对数据进行进一步的预处理，建立发音，字母和特殊字符与整数之间的映射词典，存入`data.pickle`中方便读取。

与模型相关的参数位于`./tensor_seq/params.py`中。

*`isTrain`的值为`1`时，对模型进行训练；值为`0`时，对训练好的模型进行测试，在命令行输入单词，模型会给出预测的读音；值为`2`时，会用测试数据集对训练好的模型的准确率进行测试。*

其他使用方法见注释：
```python
# Learning rate
learning_rate = 0.001
# Optimizer used by the model, 0 for SGD, 1 for Adam, 2 for RMSProp
optimizer_type = 1
# Mini-batch size
batch_size = 512
# Cell type, 0 for LSTM, 1 for GRU
Cell_type = 0
# Activation function used by RNN cell, 0 for tanh, 1 for relu, 2 for sigmoid
activation_type = 0
# Number of cells in each layer
rnn_size = 128
# Number of layers
num_layers = 2
# Embedding size for encoding part and decoding part
encoding_embedding_size = 128
decoding_embedding_size = encoding_embedding_size
# Decoder type, 0 for basic, 1 for beam search
Decoder_type = 1
# Beam width for beam search decoder
beam_width = 3
# Number of max epochs for training
epochs = 60
# 1 for training, 0 for test the already trained model, 2 for evaluate performance
isTrain = 1
# Display the result of training for every display_step
display_step = 50
# max number of model to keep
max_model_number = 5
```

设定完成后可以通过运行以下命令对模型进行训练。

```bash
python run.py
```
模型训练过程中会在`./tensor_seq`目录下生成`graph`与`model`两个文件夹，其中`model`中保存了模型训练过程中的各个状态，用于训练完成后读取模型进行发音预测。`graph`中保存了计算图的相关信息，以及training loss和validation loss，利用tensorboard工具可以对模型以及相关参数进行可视化。

输入以下命令来开启tensorboard
```bash
tensorboard --logdir ./graph/
```
开启以后只要在浏览器中输入`http://127.0.0.1:6006`即可访问tensorboard。

模型训练完成后将参数`isTrain`的值设置为0后再次运行`model.py`可以对训练好的模型进行测试。
使用方法：
```bash
python run.py
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
位于`./Split_Dataset`文件夹内的`sp.py`将`cmudict.0.7a`中的单词及读音提取出来，去除标点符号的读法，存至文件`whole`，并且随机抽取10000条数据作为测试数据(`testing`)，10000条作为验证数据(`validation`)，剩下的作为训练数据(`training`)。

#### 使用方法
```bash
cd  ./Split_Dataset
python sp.py
```
运行完成后将会在`Split_Dataset`中生成`whole`,`testing`,`validation`,`training`四个文件。

### 4. 音标转换工具
---
代码位于`./data_utils`文件夹内。由于爬虫爬取到的音标为utf-8编码格式的国际音标（[IPA](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet)），而进行发音预测时的输出为[ARPAbet](https://en.wikipedia.org/wiki/ARPABET)音标，因此如果想要使用自己爬取获得的数据进行模型的训练的话需要对数据进行预处理，将所有的IPA音标转化为ARPAbet音标。

主要使用了[`ipapy`](https://github.com/pettarin/ipapy)模块（用于处理IPA国际音标字符串，具体用法参考项目主页，暂不支持重音处理）。

####使用方法
代码具体实现时对`ipapy`进行了简单的修改，增加了æ的对应音标。

首先下载改版的`ipapy`模块
```shell
cd ./data_utils
git clone https://github.com/yuyue9284/ipapy.git
```
然后运行
```shell
python ./convert_to_arpabet.py
```
这将在`./data_utils`文件夹下生成`data`文件夹，其中`en.csv`，`us.csv`，`final_result`分别为单词与英音IPA音标对照文件，单词与美音IPA音标对照文件，以及最终符合模型输入要求的单词与ARPAbet音标文件。

`final_result`***是将英音美音对照同时输入到了一个文件之中，而在模型训练时应当仅仅使用一个口音，这样才能保证模型的准确性，因此***`final_result`***文件并不能直接当做数据集进行模型的训练，需要进行进一步的分割***。

### TODO
---
- [ ] 加入attention机制
- [ ] 采用输入逆序进行训练
