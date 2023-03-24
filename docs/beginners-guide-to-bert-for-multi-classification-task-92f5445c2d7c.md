# 多分类任务的 BERT 初学者指南

> 原文：<https://towardsdatascience.com/beginners-guide-to-bert-for-multi-classification-task-92f5445c2d7c?source=collection_archive---------3----------------------->

![](img/cabd8e8b7794512f82bd78eb8948820a.png)

Original Photo by [David Pisnoy](https://unsplash.com/@davidpisnoy?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText). It was later modified to include some inspiring quotes.

本文的目的是提供如何使用 BERT 进行多分类任务的分步指南。BERT(**B**I directional**E**n coder**R**presentations from**T**transformers)，是 Google 提出的一种新的预训练语言表示方法，旨在解决广泛的自然语言处理任务。该模型基于无监督的深度双向系统，并在 2018 年首次向公众发布时成功实现了最先进的结果。如果你想了解更多，可以在下面的[链接](https://arxiv.org/abs/1810.04805)中找到学术论文。

本教程有 5 个部分:

1.  设置和安装
2.  数据集准备
3.  培训模式
4.  预言；预测；预告
5.  结论

# [第 1 节]设置和安装

在本教程中，我将使用 Ubuntu 18.04 搭配单个 GeForce RTX 2080 Ti。就我个人而言，我不建议在没有 GPU 的情况下进行训练，因为基础模型太大了，训练时间非常长。

## 虚拟环境

建议建立一个虚拟环境。如果你是第一次使用 Ubuntu，打开终端，将目录切换到你想要的位置。它将是您环境的根文件夹。运行以下命令安装 pip:

```
sudo apt-get install python3-pip
```

然后，运行以下命令安装 virtualenv 模块:

```
sudo pip3 install virtualenv
```

您现在可以创建自己的虚拟环境(用您喜欢的任何名称替换 bertenv):

```
virtualenv bertenv
```

如果你喜欢不使用 virtualenv 模块，还有一种方法可以创建虚拟环境，只需使用 python3

```
python3 -m venv bertenv
```

您应该已经创建了一个 **bertenv** 文件夹。查看以下[链接](https://gist.github.com/Geoyi/d9fab4f609e9f75941946be45000632b)了解更多信息。您可以使用以下命令激活虚拟环境:

```
source bertenv/bin/activate
```

## 开源代码库

从下面的[链接](https://github.com/google-research/bert)克隆存储库。完成后，解压缩 zip 文件并将其放入您选择的目录中。你应该有一个 **bert-master** 文件夹。我把它放在虚拟环境文件夹旁边。因此，在根目录中，我有以下子文件夹:

1.  贝尔坦夫
2.  伯特-马斯特

## Python 模块

BERT 只需要 tensorflow 模块。您必须安装等于或高于 1.11.0 的版本。确保您安装了 CPU 版本或 GPU 版本，但不是两者都安装。

```
tensorflow >= 1.11.0   # CPU Version of TensorFlow.                       tensorflow-gpu  >= 1.11.0  # GPU version of TensorFlow.
```

你可以通过 pip 或者位于 **bert-master** 文件夹中的 [*requirement.txt*](https://github.com/google-research/bert/blob/master/requirements.txt) 文件来安装。

## 伯特模型

我们将需要一个微调过程的基础模型。我将在本教程中使用 [**BERT-Base，Cased**](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip) ( 12 层，768-hidden，12-heads，110M 参数)。如果你想试试 **BERT-Large** ( 24 层，1024 隐藏，16 头，340M 参数)，确保你有足够的内存。12GB 的 GPU 不足以运行 BERT-Large。我个人会推荐你用 64GB 的 GPU 做 BERT-Large。在撰写本文时，BERT 背后的团队还发布了其他模型，如中文、多语言和全词屏蔽。请通过以下[链接](https://github.com/google-research/bert#pre-trained-models)查看。下载完文件后，解压缩该文件，您应该得到以下文件:

1.  三个 ckpt 文件
2.  vocab.txt
3.  伯特配置文件

将它们放在**模型**文件夹中，并将其移动到 **bert-master** 文件夹中。请转到下一节数据集准备。

# [第 2 节]数据集准备

对于 BERT 来说，数据准备是非常复杂的，因为官方的 github 链接并没有包含太多需要什么样的数据。首先，有 4 个类可用于序列分类任务:

1.  Xnli(跨语言 nli)
2.  多体裁自然语言推理
3.  微软研究释义语料库
4.  语言可接受性语料库

所有的类都是基于数据处理器类(参见第 177 行的[*run _ classifier . py*](https://github.com/google-research/bert/blob/master/run_classifier.py)文件)，该类用于将数据提取到以下内容:

1.  **guid** :示例的唯一 id。
2.  **text_a** :字符串数据。第一个序列的未标记文本。对于单序列任务，只能指定此序列。
3.  **text_b** :(可选)字符串数据。第二个序列的未标记文本。必须为序列对任务指定 Only。
4.  **标签**:字符串数据。示例的标签。这应该为训练和评估示例指定，而不是为测试示例指定。

换句话说，我们可以修改我们的数据集来模拟 4 个类之一的模式和格式，或者编写我们自己的类来扩展 DataProcessor 类以读取我们的数据。在本教程中，我将把数据集转换成可乐格式，因为它是所有格式中最简单的。其他数据集的例子可以在下面的[链接](https://gluebenchmark.com/tasks)(胶水版)中找到。

在 BERT 的原始版本中，[*run _ classifier . py*](https://github.com/google-research/bert/blob/master/run_classifier.py)基于从三个 tsv 文件中读取输入:

1.  train.tsv(无标题)
2.  dev.tsv(评估，无标题)
3.  test.tsv(头是必需的)

对于 *train.tsv* 和 *dev.tsv* ，你应该有以下格式(无表头):

```
a550d    1    a    To clarify, I didn't delete these pages.
kcd12    0    a    Dear god this site is horrible.
7379b    1    a    I think this is not appropriate.
cccfd    2    a    The title is fine as it is.
```

1.  **列 1** :示例的 guid。它可以是任何唯一的标识符。
2.  **第 2 列**:示例的标签。它是基于字符串的，可以是文本形式，而不仅仅是数字。为了简单起见，我在这里只使用数字。
3.  **第 3 列**:第二个序列的未分词文本。必须为序列对任务指定 Only。因为我们正在进行单序列任务，所以这只是一个一次性的专栏。对于所有的行，我们都用“a”来填充它。
4.  **第 4 列**:第一个序列的未分词文本。用示例的文本填充它。

对于 *test.tsv* ，你应该有如下格式(需要表头):

```
guid     text
casd4    I am not going to buy this useless stuff.
3ndf9    I wanna be the very best, like no one ever was
```

## 将其他来源的数据转换成所需的格式

如果您有不同于上面给出的格式的数据，您可以使用 pandas 模块和 sklearn 模块轻松地转换它。通过 pip 安装模块(确保虚拟环境已激活):

```
pip install pandas
```

如果您打算使用 train_test_split，也要安装 sklearn:

```
pip install sklearn
```

例如，如果我们有以下 csv 格式的训练数据集:

```
id,text,label
sadcc,This is not what I want.,1
cj1ne,He seriously have no idea what it is all about,0
123nj,I don't think that we have any right to judge others,2
```

我们可以使用以下代码轻松加载数据集并将其转换为相应的格式(相应地修改路径):

## 从 csv 文件创建数据帧

```
import pandas as pddf_train = pd.read_csv('dataset/train.csv')
```

## 从现有数据框架创建新的数据框架

```
df_bert = pd.DataFrame({'guid': df_train['id'],
    'label': df_train['label'],
    **'alpha': ['a']*df_train.shape[0]**,
    'text': df_train['text']})
```

粗体突出显示的部分意味着我们将根据 df_train 数据帧中的行数用字符串 a 填充列 alpha。shape[0]指的是行数，而 shape[1]指的是列数。

## 输出 tsv 文件

```
df_bert_train.to_csv('dataset/train.tsv', **sep='\t'**, index=False, header=False)
```

不要对 to_csv 函数调用感到惊讶，因为除了分隔符之外，tsv 和 csv 具有相似的格式。换句话说，我们只需要提供正确的制表符分隔符，它就会变成一个 tsv 文件(以粗体突出显示)。

下面是创建所有必需文件的完整工作代码片段(相应地修改路径)。

```
import pandas as pd
from sklearn.model_selection import train_test_split#read source data from csv file
df_train = pd.read_csv('dataset/train.csv')
df_test = pd.read_csv('dataset/test.csv')#create a new dataframe for train, dev data
df_bert = pd.DataFrame({'guid': df_train['id'],
    'label': df_train['label'],
    'alpha': ['a']*df_train.shape[0],
    'text': df_train['text']})#split into test, dev
df_bert_train, df_bert_dev = train_test_split(df_bert, test_size=0.01)#create new dataframe for test data
df_bert_test = pd.DataFrame({'guid': df_test['id'],
    'text': df_test['text']})#output tsv file, no header for train and dev
df_bert_train.to_csv('dataset/train.tsv', sep='\t', index=False, header=False)
df_bert_dev.to_csv('dataset/dev.tsv', sep='\t', index=False, header=False)
df_bert_test.to_csv('dataset/test.tsv', sep='\t', index=False, header=True)
```

一旦你有了所有需要的文件，将**数据集**文件夹移动到 **bert-master** 文件夹。让我们继续下一部分来微调您的模型。

# [第三节]培训模式

微调 BERT 模型最简单的方法是通过命令行(终端)运行 run_classifier.py。在此之前，我们需要根据我们的标签修改 python 文件。最初的版本是使用 0 和 1 作为标签的二进制分类。如果使用不同的标签进行多分类或二分类，需要更改 **ColaProcessor** 类的 *get_labels()* 函数(第 354 行，如果使用其他**数据处理器**类，请相应修改):

## 原始代码

```
def get_labels(self):
    return ["0", "1"]
```

## 5 标签多分类任务

```
def get_labels(self):
    return ["0", "1", "2", "3", "4"]
```

## 使用不同标签的二元分类

```
def get_labels(self):
    return ["POSITIVE", "NEGATIVE"]
```

如果您遇到如下任何键错误，这意味着您的 get_labels 函数与您的数据集不匹配:

```
label_id = label_map[example.label]
KeyError: '2'`
```

我们现在已经准备好接受训练了。如果您使用的是 NVIDIA GPU，您可以在终端中键入以下内容来检查状态和 CUDA 版本。

```
nvidia-smi
```

更改目录指向 **bert-master** 文件夹，确保**数据集**文件夹和 **bert-master** 文件夹中的所需文件。建议通过命令行运行培训，而不是使用 jupyter notebook，原因如下:

1.  官方代码使用 2 个单位缩进，不同于笔记本默认的 4 个单位缩进。
2.  内存问题和配置用于训练的 GPU 的额外代码。

## 因素

在此之前，让我们探索一下可以针对培训流程进行微调的参数:

1.  **data_dir** :包含 train.tsv、dev.tsv、test.tsv 的输入目录。
2.  **bert_config_file** :预先训练好的 bert 模型对应的 config json 文件。这指定了模型架构。
3.  **任务名称**:训练任务的名称。4 个选项可用(xnli、mrpc、mnli、cola)。
4.  **vocab_file** :训练 BERT 模型的词汇文件。
5.  **output_dir** :模型检查点将被写入的输出目录。
6.  **init_checkpoint** :初始检查点(通常来自预训练的 BERT 模型)。
7.  **do_lower_case** :输入的文本是否小写。对于无大小写应为 True，对于有大小写应为 False。
8.  **max_seq_length** :分词后最大总输入序列长度。长于此长度的序列将被截断，短于此长度的序列将被填充。默认值为 128。
9.  **do_train** :是否在 train.tsv 上运行训练。
10.  **do_eval** :是否在 dev.tsv 上运行评估
11.  **do_predict** :是否在 test.tsv 上以推理模式运行模型
12.  **train_batch_size** :训练的总批量。默认值为 32。
13.  **eval_batch_size** :评估的总批量。默认值为 8
14.  **predict_batch_size** :测试和预测的总批量。默认值为 8。
15.  **learning _ rate**:Adam 的初始学习率。默认为 5e-5。
16.  **num_train_epochs** :要执行的训练总次数。默认值为 3.0。
17.  **热身 _ 比例**:从 0 到 1 进行线性学习率热身的训练比例。默认值为 0.1 表示 10%。
18.  **save_checkpoints_steps** :保存模型检查点的步数间隔。默认值为 1000。
19.  **iterations_per_loop** :每次估算器调用的步数间隔。默认值为 1000。
20.  **使用 _tpu** :是否使用 tpu。
21.  **tpu _ 姓名**:云 TPU 用于训练。
22.  **tpu 区**:TPU 云所在的 GCE 区。
23.  **gcp_project** :启用云 TPU 的项目的项目名称
24.  **主** : TensorFlow 主 URL。
25.  **数量 _ TPU _ 核心数**:仅当**使用 _tpu** 为真时使用。要使用的 TPU 核心总数。

不要被参数的数量所淹没，因为我们不会指定每个参数。

## 培训说明

从[官方文档](https://github.com/google-research/bert)中，它建议通过以下命令行调用将路径导出为变量(如果您使用的是 Windows 操作系统，请将导出替换为设置):

```
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
```

在本教程中，我将不导出路径，因为您仍然需要在命令行中指定它。只要确保你正确地组织了你的文件夹，你就可以开始了。要指定 GPU，您需要在 python 调用之前键入它(示例如下，不要运行它):

```
CUDA_VISIBLE_DEVICES=0 python script.py
```

0 是指 GPU 的顺序。使用以下命令检查它:

```
nvidia-smi
```

在 **bert-master** 文件夹中，创建一个输出文件夹。我就把它叫做 **bert_output** 。确保在 **bert-master** 文件夹中有以下文件夹和文件:

1.  **数据集**文件夹(包含 train.tsv，dev.tsv，test.tsv)
2.  **型号**文件夹(包含 ckpt，vocab.txt，bert_config.json)
3.  **bert_output** 文件夹(空)

## 通过命令行培训

确保终端指向 bert-master 目录，并且虚拟环境已激活。根据您的偏好修改参数并运行它。我做了以下更改:

1.  将 **train_batch_size** 减少到 2:如果你有足够的内存，可以随意增加。这影响了训练时间。越高，训练时间越短。
2.  将**的保存 _ 检查点 _ 步骤**增加到 10000。我不想有这么多检查点，因为每个检查点模型都是原始大小的 3 倍。放心，这个脚本一次只保留 5 个模型。旧型号将被自动删除。强烈建议将其保持为 1000(默认值)。
3.  将**最大序列长度**减少到 64。由于我的数据集 99%的长度不超过 64，将其设置得更高是多余的。根据数据集对此进行相应的修改。默认值为 128。(序列长度是指词块标记化后的字符长度，请考虑这一点)。

```
CUDA_VISIBLE_DEVICES=0 python run_classifier.py --task_name=cola --do_train=true --do_eval=true --data_dir=./dataset --vocab_file=./model/vocab.txt --bert_config_file=./model/bert_config.json --init_checkpoint=./model/bert_model.ckpt --max_seq_length=64 --train_batch_size=2 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=./bert_output/ --do_lower_case=False --save_checkpoints_steps 10000
```

训练应该已经开始，显示跑步步数/秒。检查 bert_output 文件夹，您应该注意到以下内容:

1.  三个 ckpt 文件(模型)
2.  tf _ 记录
3.  检查点和事件文件(临时文件，可以在培训后安全地忽略和删除)
4.  graph.pbtxt

这可能需要几个小时到几天的时间，具体取决于您的数据集和配置。

## 培训完成

培训完成后，您应该有一个 *eval_results.txt* 来指示您的模型的性能。

```
eval_accuracy = 0.96741855
eval_loss = 0.17597112
global_step = 236962
loss = 0.17553209
```

确定模型的最高步骤数。如果您对此不确定，请在文本编辑器中打开检查点，您应该会看到以下内容:

```
model_checkpoint_path: "model.ckpt-236962"
all_model_checkpoint_paths: "model.ckpt-198000"
all_model_checkpoint_paths: "model.ckpt-208000"
all_model_checkpoint_paths: "model.ckpt-218000"
all_model_checkpoint_paths: "model.ckpt-228000"
all_model_checkpoint_paths: "model.ckpt-236962"
```

在这种情况下，最高步骤是 236962。我们现在可以用这个模型来预测结果。

# [第四节]预测

为了进行预测，我们将使用相同的 *run_classifier.py* 。这一次，我们需要将 **do_predict** 指定为 True，并将 **init_checkpoint** 设置为我们拥有的最新模型 model.ckpt-236962(根据您拥有的最高步骤进行相应修改)。但是，您需要确保 **max_seq_length** 与您用于训练的相同。

```
CUDA_VISIBLE_DEVICES=0 python run_classifier.py --task_name=cola --do_predict=true --data_dir=./dataset --vocab_file=./model/vocab.txt --bert_config_file=./model/bert_config.json --init_checkpoint=./bert_output/model.ckpt-236962 --max_seq_length=64 --output_dir=./bert_output/
```

一旦流程完成，您应该在 **bert_output** 文件夹中有一个 test_results.tsv(取决于您为 **output_dir** 指定的内容)。如果使用文本编辑器打开它，您应该会看到以下输出:

```
1.4509245e-05 1.2467547e-05 0.99994636
1.4016414e-05 0.99992466 1.5453812e-05
1.1929651e-05 0.99995375 6.324972e-06
3.1922486e-05 0.9999423 5.038059e-06
1.9996814e-05 0.99989235 7.255715e-06
4.146e-05 0.9999349 5.270801e-06
```

列的数量取决于标签的数量。每一列按照您为 get_labels()函数指定的顺序表示每个标签。该值代表预测的可能性。例如，模型预测第一个示例属于第 3 类，因为它具有最高的概率。

## 将结果映射到相应的类

如果您想映射结果来计算精确度，可以使用以下代码(相应地修改):

```
import pandas as pd#read the original test data for the text and id
df_test = pd.read_csv('dataset/test.tsv', sep='\t') #read the results data for the probabilities
df_result = pd.read_csv('bert_output/test_results.tsv', sep='\t', header=None)#create a new dataframe
df_map_result = pd.DataFrame({'guid': df_test['guid'],
    'text': df_test['text'],
    'label': df_result.idxmax(axis=1)})#view sample rows of the newly created dataframe
df_map_result.sample(10)
```

**idxmax** 是一个函数，用于返回请求轴上第一次出现的最大值的索引。不包括 NA/null 值。在这种情况下，我们传递 1 作为轴来表示列而不是行。

# [第五节]结论

在本教程中，我们已经学会了针对多分类任务微调 BERT。供您参考，BERT 可以用于其他自然语言处理任务，而不仅仅是分类。就我个人而言，我也测试了基于 BERT 的中文情感分析，结果出乎意料的好。请记住，非拉丁语言(如中文和朝鲜语)是字符标记化，而不是单词标记化。请随意在不同种类的数据集上尝试其他模型。感谢您的阅读，祝您有美好的一天。下篇再见！

# 伯特相关的故事

1.  [优化和导出在线服务 BERT 模型的 3 种方法](/3-ways-to-optimize-and-export-bert-model-for-online-serving-8f49d774a501)

# 参考

1.  【https://arxiv.org/abs/1810.04805 
2.  [https://github.com/google-research/bert](https://github.com/google-research/bert)
3.  [https://gist . github . com/geo yi/d 9 fab 4 f 609 e 9 f 75941946 be 45000632 b](https://gist.github.com/Geoyi/d9fab4f609e9f75941946be45000632b)
4.  [https://gluebenchmark.com/tasks](https://gluebenchmark.com/tasks)
5.  [https://github . com/Google-research/Bert/blob/master/run _ classifier . py](https://github.com/google-research/bert/blob/master/run_classifier.py)
6.  [https://MC . ai/a-guide-to-simple-text-class ification-with-Bert/](https://mc.ai/a-guide-to-simple-text-classification-with-bert/)
7.  [https://appliedmachinehlearning . blog/2019/03/04/stat-of-art-text-classification-using-Bert-model-predict-the-happy-hackere earth-challenge/](https://appliedmachinelearning.blog/2019/03/04/state-of-the-art-text-classification-using-bert-model-predict-the-happiness-hackerearth-challenge/)