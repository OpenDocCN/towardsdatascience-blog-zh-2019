# Keras & Tensorflow 中基于深度卷积网络的模因文本生成

> 原文：<https://towardsdatascience.com/meme-text-generation-with-a-deep-convolutional-network-in-keras-tensorflow-a57c6f218e85?source=collection_archive---------6----------------------->

这篇文章的目标是描述如何为文本生成构建一个深度的 conv 网络，但是比我读过的一些现有文章更有深度的 T2。这将是一个实用的指南，虽然我建议了许多最佳实践，但我不是深度学习理论的专家，也没有读过每一篇相关的研究论文。我将讨论数据清理、训练、模型设计和预测算法。

# 步骤 1:构建培训数据

我们将从中提取的原始数据集是来自 [Imgflip meme 生成器](https://imgflip.com/memegenerator)用户的大约 1 亿条公共 Meme 说明。为了加快训练速度和降低模型的复杂性，我们只使用 48 个最流行的模因和每个模因正好 20，000 个标题，总共 960，000 个标题作为训练数据。然而，由于我们正在构建一个世代模型，因此标题中的每个*字符*都将有一个训练示例，总计约 45，000，000 个训练示例。这里选择了字符级生成而不是单词级，因为模因倾向于使用拼写和语法…呃…创造性地。此外，字符级深度学习是单词级深度学习的超集，因此如果你有足够的数据，并且你的模型设计足以学习所有的复杂性，则可以实现更高的准确性。如果你尝试下面的成品模型，你也会发现 char 级别会更有趣！

如果第一个模因标题是“制造所有模因”，则训练数据看起来如下。我省略了从数据库读取和执行初始清理的代码，因为它非常标准，并且可以通过多种方式完成。

![](img/fe9e52026a61048d11a1150aaf8084a3.png)

```
training_data = [
    ["000000061533  0  ", "m"],
    ["000000061533  0  m", "a"],
    ["000000061533  0  ma", "k"],
    ["000000061533  0  mak", "e"],
    ["000000061533  0  make", "|"],
    ["000000061533  1  make|", "a"],
    ["000000061533  1  make|a", "l"],
    ["000000061533  1  make|al", "l"],
    ["000000061533  1  make|all", " "],
    ["000000061533  1  make|all ", "t"],
    ["000000061533  1  make|all t", "h"],
    ["000000061533  1  make|all th", "e"],
    ["000000061533  1  make|all the", " "],
    ["000000061533  1  make|all the ", "m"],
    ["000000061533  1  make|all the m", "e"],
    ["000000061533  1  make|all the me", "m"],
    ["000000061533  1  make|all the mem", "e"],
    ["000000061533  1  make|all the meme", "s"],
    ["000000061533  1  make|all the memes", "|"],

    ... 45 million more rows here ...
]
# we'll need our feature text and labels as separate arrays later
texts = [row[0] for row in training_data]
labels = [row[1] for row in training_data]
```

和机器学习中的大部分事情一样，这只是一个分类问题。我们将左边的文本字符串分类到大约 70 个不同的桶中，桶是字符。

我们来解一下格式。

*   前 12 个字符是 meme 模板 ID。这使得模型能够区分我们正在喂它的 48 种不同的迷因。该字符串用零填充，因此所有 id 的长度相同。
*   `0`或`1`是当前被预测文本框的索引，一般 0 是顶框，1 是底框，虽然[很多模因更复杂](https://imgflip.com/memegenerator/Distracted-Boyfriend)。这两个空格只是额外的间隔，以确保模型可以将框索引与模板 ID 和 meme 文本区分开来。注意:关键是我们的卷积核宽度(见后文)不超过 4 个空格加上索引字符，也就是≤ 5。
*   之后是迄今为止迷因的文本，用`|`作为文本框的结束字符。
*   最后，最后一个字符本身(第二个数组项)是序列中的下一个字符。

在训练之前，对数据使用了几种清理技术:

*   修剪开头和结尾的空白，用一个空格字符替换重复的空白(`\s+`)。
*   应用最小长度为 10 个字符的字符串，这样我们就不会产生无聊的一个单词或一个字母的模因。
*   应用 82 个字符的最大字符串长度，这样我们就不会生成超长的模因，因为模型将训练得更快。82 是任意的，它只是使整体训练字符串大约 100 个字符。
*   将所有内容都转换成小写，以减少模型必须学习的字符数量，因为许多模因都是大写字母。
*   跳过带有非 ascii 字符的 meme 标题，以降低模型必须学习的复杂性。这意味着我们的特征文本和标签都将来自大约 70 个字符的集合，这取决于训练数据恰好包括哪些 ascii 字符。
*   跳过包含管道字符`|`的 meme 标题，因为它是我们特殊的文本框结尾字符。
*   通过语言检测库运行文本，跳过不太可能是英语的迷因说明。提高了我们生成的文本的质量，因为模型只需学习一种语言，而相同的字符序列在多种语言中可以有不同的含义。
*   跳过我们已经添加到训练集的重复迷因标题，以减少模型简单记忆整个迷因标题的机会。

我们的数据现在可以输入神经网络了！

# 步骤 2:数据张量化

这可能是也可能不是一个词。有趣的事实:显然，我们在深度学习领域是异教徒，因为我们称多维数组为张量，而不是数学术语“全息”，这是一种不需要特殊变换属性的广义张量。但无论如何，张量听起来比全息更酷；)

首先，这是我们下面要做的所有事情的 python 导入代码:

```
**from** keras **import** Sequential
**from** keras.preprocessing.sequence **import** pad_sequences
**from** keras.callbacks **import** ModelCheckpoint
**from** keras.layers **import** Dense, Dropout, GlobalMaxPooling1D, Conv1D, MaxPooling1D, Embedding
**from** keras.layers.normalization **import** BatchNormalization
**import** numpy **as** np
**import** util  # util is a custom file I wrote, see github link below
```

神经网络对数字的张量(向量/矩阵/多维数组)进行操作，因此我们需要相应地重构我们的文本。我们的每个训练文本都将被转换为一个整数数组(一个秩为 1 的张量)，方法是用数据中找到的大约 70 个唯一字符的数组中的相应索引来替换每个字符。字符数组的顺序是任意的，但是我们选择按字符频率来排序，这样当改变训练数据量时，它会保持大致一致。Keras 有一个 Tokenizer 类，你可以用它来做这件事(char_level=True ),但是我写了自己的 util 函数，因为它们肯定比 Keras tokenizer 快。

```
# output: {' ': 1, '0': 2, 'e': 3, ... }
char_to_int = util.map_char_to_int(texts)# output: [[2, 2, 27, 11, ...], ... ]
sequences = util.texts_to_sequences(texts, char_to_int)labels = [char_to_int[char] for char in labels]
```

这些是我们的数据按频率顺序包含的字符:

```
0etoains|rhl1udmy2cg4p53wf6b897kv."!?j:x,*"z-q/&$)(#%+_@=>;<][~`^}{\
```

接下来，我们将使用前导零填充整数序列，使它们都具有相同的长度，因为模型的张量数学要求每个训练示例的形状都相同。(*注意:我可以在这里使用低至 100 的长度，因为我们的文本只有 100 个字符，但是我希望以后所有的池操作都能被 2 整除。*)

```
SEQUENCE_LENGTH = 128
data = pad_sequences(sequences, maxlen=SEQUENCE_LENGTH)
```

最后，我们将整理我们的训练数据，并将其分成训练集和验证集。洗牌(使顺序随机化)确保数据的特定子集不总是我们用来验证准确性的子集。将一些数据拆分到一个验证集中，可以让我们衡量模型在我们不允许它用于训练的例子上的表现。

```
*# randomize order of training data*
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]*# validation set can be much smaller if we use a lot of data* validation_ratio = 0.2 ifdata.shape[0] < 1000000 else0.02
num_validation_samples = int(validation_ratio * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]
```

# 第三步:模型设计

![](img/86c1b0c5c1de235c3439ef862fd64d79.png)

我选择使用卷积网络，因为卷积训练简单快速。我确实短暂地测试了一个两层 LSTM，但每次训练的精度比 conv 网络差，即使是这么小的 LSTM，预测的时间也比宇宙的年龄长(好吧，可能只是感觉太长了)。生成敌对网络是具有巨大潜力的美丽生物，但将它们用于文本生成仍处于早期阶段，我的第一次尝试也乏善可陈。也许这将是我的下一篇文章…

好了，下面是在 Keras 中用来构建我们的 conv 网模型的代码:

```
EMBEDDING_DIM = 16model = Sequential()model.add(Embedding(len(char_to_int) + 1, EMBEDDING_DIM, input_length=SEQUENCE_LENGTH))
model.add(Conv1D(1024, 5, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Dropout(0.25))
model.add(Conv1D(1024, 5, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Dropout(0.25))
model.add(Conv1D(1024, 5, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Dropout(0.25))
model.add(Conv1D(1024, 5, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Dropout(0.25))
model.add(Conv1D(1024, 5, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(len(labels_index), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
```

那里发生了很多事情。下面是所有代码正在做的事情:

首先，该模型使用 Keras 嵌入将每个输入示例从 128 个整数的数组(每个表示一个文本字符)转换为 128x16 的矩阵。嵌入是一个层，它学习一种最佳方式来将我们的每个字符从表示为整数转换为表示为类似于`[0.02, ..., -0.91]`的 16 个浮点数的数组。这使得模型可以通过在 16 维空间中将字符相互靠近嵌入来了解哪些字符被类似地使用，并最终提高模型预测的准确性。

接下来，我们添加 5 个卷积层，每个卷积层的内核大小为 5，1024 个滤波器，以及一个 ReLU 激活。从概念上讲，第一个 conv 层正在学习如何从字符构造单词，后面的几层正在学习构造更长的单词和单词链(n-grams)，每一层都比前一层更抽象。

*   `padding='same'`用于确保层的输出尺寸与输入尺寸相同，否则宽度为 5 的卷积会使内核每侧的层尺寸减少 2。
*   选择 1024 作为过滤器的数量，因为它是训练速度和模型精度之间的良好折衷，通过反复试验来确定。对于其他数据集，我建议从 128 个过滤器开始，然后增加/减少两倍，看看会发生什么。更多的过滤器通常意味着更好的模型准确性，但是训练更慢，运行时预测更慢，并且模型更大。但是，如果您的数据太少或过滤器太多，您的模型可能会过拟合，精度会直线下降，在这种情况下，您应该减少过滤器。
*   在测试了 2、3、5 和 7 之后，选择了内核大小为 5。内核 2 和 3 的表现更差，内核 7 与内核 2 和内核 3 相似，但速度更慢，因为需要多训练 7/5 个参数。在我的研究中，其他人已经成功地在各种组合中使用了大小为 3 到 7 的内核，但我的观点是，大小为 5 的内核通常在文本数据上表现不错，并且您可以随时在以后进行实验，为您的特定数据集挤出更多的准确性。
*   选择 ReLU 激活是因为它快速、简单，并且非常适合各种各样的用例。我从阅读一些文章和研究论文中得出的结论是，Leaky ReLU 或其他变体可能会对一些数据集产生轻微的改善，但并不保证会更好，而且在较大的数据集上不太可能明显。
*   在每个 conv 图层后添加批次归一化，以便基于给定批次的均值和方差对下一图层的输入参数进行归一化。深度学习工程师还没有完全理解这种机制，但我们知道标准化输入参数可以提高训练速度，并且由于消失/爆炸梯度，这对于更深的网络变得更加重要。[原批次归一化论文](https://arxiv.org/abs/1502.03167)成绩斐然。
*   在每个 conv 图层后添加一点点遗漏，以帮助防止图层简单地记忆数据和过度拟合。Dropout(0.25)随机删除 25%的参数(将其设置为零)。
*   MaxPooling1D(2)被添加到每个 conv 层之间，以将我们的 128 个字符的序列对半“挤压”成后续层中的 64、32、16 和 8 个字符的序列。从概念上讲，这允许卷积过滤器从更深层的文本中学习更抽象的模式，因为我们的 width 5 内核在通过每个 max pooling 操作将维度减少 2 倍后将跨越两倍的字符。

在所有 conv 层之后，我们使用一个全局最大池层，它与普通最大池层相同，只是它会自动选择缩小输入大小的程度，以匹配下一层的大小。最后的层只是具有 1024 个神经元的标准密集(全连接)层，最后是 70 个神经元，因为我们的分类器需要输出 70 个不同标签中每个标签的概率。

model.compile 步骤非常标准。RMSprop 优化器是一个相当不错的全面优化器，我没有尝试为这个神经网络改变它。`loss=sparse_categorical_crossentropy`告诉模型，我们希望它进行优化，从一组 2 个或更多类别(也称为标签)中选择最佳类别。“稀疏”部分指的是我们的标签是 0 到 70 之间的整数，而不是每个长度为 70 的独热数组。对标签使用一个热阵列会占用更多的内存、更多的处理时间，并且不会影响模型的准确性。不要使用一个热标签！

Keras 有一个很好的`model.summary()`函数，可以让我们查看我们的模型:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, 128, 16)           1136
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 128, 1024)         82944
_________________________________________________________________
batch_normalization_1 (Batch (None, 128, 1024)         4096
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 64, 1024)          0
_________________________________________________________________
dropout_1 (Dropout)          (None, 64, 1024)          0
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 64, 1024)          5243904
_________________________________________________________________
batch_normalization_2 (Batch (None, 64, 1024)          4096
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, 32, 1024)          0
_________________________________________________________________
dropout_2 (Dropout)          (None, 32, 1024)          0
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 32, 1024)          5243904
_________________________________________________________________
batch_normalization_3 (Batch (None, 32, 1024)          4096
_________________________________________________________________
max_pooling1d_3 (MaxPooling1 (None, 16, 1024)          0
_________________________________________________________________
dropout_3 (Dropout)          (None, 16, 1024)          0
_________________________________________________________________
conv1d_4 (Conv1D)            (None, 16, 1024)          5243904
_________________________________________________________________
batch_normalization_4 (Batch (None, 16, 1024)          4096
_________________________________________________________________
max_pooling1d_4 (MaxPooling1 (None, 8, 1024)           0
_________________________________________________________________
dropout_4 (Dropout)          (None, 8, 1024)           0
_________________________________________________________________
conv1d_5 (Conv1D)            (None, 8, 1024)           5243904
_________________________________________________________________
batch_normalization_5 (Batch (None, 8, 1024)           4096
_________________________________________________________________
global_max_pooling1d_1 (Glob (None, 1024)              0
_________________________________________________________________
dropout_5 (Dropout)          (None, 1024)              0
_________________________________________________________________
dense_1 (Dense)              (None, 1024)              1049600
_________________________________________________________________
batch_normalization_6 (Batch (None, 1024)              4096
_________________________________________________________________
dropout_6 (Dropout)          (None, 1024)              0
_________________________________________________________________
dense_2 (Dense)              (None, 70)                71750
=================================================================
Total params: 22,205,622
Trainable params: 22,193,334
Non-trainable params: 12,288
_________________________________________________________________
```

如果你不喜欢在脑子里做张量形状乘法，参数计数会特别有用。当调整我们上面讨论的超参数时，关注模型的参数计数是有用的，它粗略地表示了模型的学习能力总量。

# 第四步:培训

现在，我们将让模型进行训练，并使用“检查点”来保存沿途的历史和最佳模型，以便我们可以在训练期间的任何时间点使用最新的模型来检查进度和进行预测。

```
*# the path where you want to save all of this model's files* MODEL_PATH = '/home/ubuntu/imgflip/models/conv_model'
*# just make this large since you can stop training at any time* NUM_EPOCHS = 48
*# batch size below 256 will reduce training speed since* # CPU (non-GPU) work must be done between each batch
BATCH_SIZE = 256*# callback to save the model whenever validation loss improves* checkpointer = ModelCheckpoint(filepath=MODEL_PATH + '/model.h5', verbose=1, save_best_only=True)*# custom callback to save history and plots after each epoch* history_checkpointer = util.SaveHistoryCheckpoint(MODEL_PATH)*# the main training function where all the magic happens!* history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=[checkpointer, history_checkpointer])
```

你只需要坐在这里，看着这个神奇的数字在几个小时内上升…

```
Train on 44274928 samples, validate on 903569 samples
Epoch 1/48
44274928/44274928 [==============================] - 16756s 378us/step - loss: 1.5516 - acc: 0.5443 - val_loss: 1.3723 - val_acc: 0.5891Epoch 00001: val_loss improved from inf to 1.37226, saving model to /home/ubuntu/imgflip/models/gen_2019_04_04_03_28_00/model.h5
Epoch 2/48
44274928/44274928 [==============================] - 16767s 379us/step - loss: 1.4424 - acc: 0.5748 - val_loss: 1.3416 - val_acc: 0.5979Epoch 00002: val_loss improved from 1.37226 to 1.34157, saving model to /home/ubuntu/imgflip/models/gen_2019_04_04_03_28_00/model.h5
Epoch 3/48
44274928/44274928 [==============================] - 16798s 379us/step - loss: 1.4192 - acc: 0.5815 - val_loss: 1.3239 - val_acc: 0.6036Epoch 00003: val_loss improved from 1.34157 to 1.32394, saving model to /home/ubuntu/imgflip/models/gen_2019_04_04_03_28_00/model.h5
Epoch 4/48
44274928/44274928 [==============================] - 16798s 379us/step - loss: 1.4015 - acc: 0.5857 - val_loss: 1.3127 - val_acc: 0.6055Epoch 00004: val_loss improved from 1.32394 to 1.31274, saving model to /home/ubuntu/imgflip/models/gen_2019_04_04_03_28_00/model.h5
Epoch 5/48
 1177344/44274928 [..............................] - ETA: 4:31:59 - loss: 1.3993 - acc: 0.5869
```

快进一下，下面是我们在每个时期的损失和准确性的一些闪亮的图表:

![](img/e7a7ab4b157b6d35bc9548c7386d3e13.png)

我发现，当训练损失/准确性比验证损失/准确性更糟糕时，这是模型学习良好而不是过度拟合的迹象。

顺便提一下，如果你使用 AWS 服务器进行培训，我发现最佳实例是 p3.2xlarge。这使用了他们截至 2019 年 4 月的最快 GPU(Tesla V100)，该实例只有一个 GPU，因为我们的模型无法非常有效地利用多个 GPU。我确实尝试过使用 Keras 的 multi_gpu_model，但它需要使批处理大小更大才能真正实现速度提升，这可能会扰乱模型的收敛能力，即使使用 4 个 gpu，它也只能快 2 倍。4 个 GPU 的 p3.8xlarge 价格是 4 倍多，所以对我来说不值得。

# 第五步:预测

好了，现在我们有了一个模型，可以输出哪个字符应该出现在迷因标题中的概率，但是我们如何使用它从头开始创建一个完整的迷因标题呢？

基本前提是，我们用我们想要为其生成文本的任何迷因来初始化一个字符串，然后我们为每个字符调用一次`model.predict`，直到模型输出框尾文本字符`|`的次数与迷因中的文本框一样多。对于上面看到的“X All The Y”迷因，文本框的默认数量是 2，我们的初始文本应该是:

```
"000000061533  0  "
```

鉴于模型输出的概率为 70，我尝试了几种不同的方法来选择下一个角色:

1.  每次选择得分最高的角色。这非常无聊，因为它每次都为给定的模因选择完全相同的文本，并且在模因中反复使用相同的单词。它一遍又一遍地吐槽“当你发现你的朋友是 X 所有 Y 迷因的最佳聚会”。它也喜欢在其他迷因中大量使用“最好的”和“派对”这样的词。
2.  给每个角色一个被选中的概率，这个概率等于模型给它的分数，但前提是这个分数高于某个阈值(最高分的 10%对这个模型来说很好)。这意味着可以选择多个字符，但偏向于得分较高的字符。这种方法成功地增加了多样性，但较长的短语有时缺乏连贯性。这里有一个来自 Futurama Fry meme 的:“不知道她是说还是只是把我的一天放了”。
3.  给每个角色一个相等的被选中的概率，但前提是它的分数足够高(最高分的 10%以上对这个模型效果很好)。此外，使用波束搜索在任何给定时间保持 N 个文本的运行列表，并使用所有字符得分的乘积，而不仅仅是最后一个字符的得分。这需要 N 倍的时间来计算，但在某些情况下似乎可以提高句子的连贯性。

我目前使用方法#2，因为它比波束搜索快得多，而且两种方法都给出了不错的结果。以下是一些随机的例子:

![](img/6f3a3bce27a8d557555f07b15f3aa3ae.png)

你可以自己玩最新的模型，从 imgflip.com/ai-meme 48 个迷因中的任何一个中产生。

下面是使用方法#2 进行运行时预测的代码。Github 上的完整实现是一个通用的波束搜索算法，因此只需将波束宽度增加到 1 以上，就可以启用波束搜索。

```
# min score as percentage of the maximum score, not absolute
MIN_SCORE = 0.1
int_to_char = {v: k for k, v in char_to_int.items()}def predict_meme_text(template_id, num_boxes, init_text = ''):
  template_id = str(template_id).zfill(12)
  final_text = ''
  for char_count in range(len(init_text), SEQUENCE_LENGTH):
    box_index = str(final_text.count('|'))
    texts = [template_id + '  ' + box_index + '  ' + final_text]
    sequences = util.texts_to_sequences(texts, char_to_int)
    data = pad_sequences(sequences, maxlen=SEQUENCE_LENGTH)
    predictions_list = model.predict(data)
    predictions = []
    for j in range(0, len(predictions_list[0])):
      predictions.append({
        'text': final_text + int_to_char[j],
        'score': predictions_list[0][j]
      })
    predictions = sorted(predictions, key=lambda p: p['score'], reverse=True)
    top_predictions = []
    top_score = predictions[0]['score']
    rand_int = random.randint(int(MIN_SCORE * 1000), 1000)
    for prediction in predictions:
      # give each char a chance of being chosen based on its score
      if prediction['score'] >= rand_int / 1000 * top_score:
        top_predictions.append(prediction)
    random.shuffle(top_predictions)
    final_text = top_predictions[0]['text']
    if char_count >= SEQUENCE_LENGTH - 1 or final_text.count('|') == num_boxes - 1:
      return final_text
```

你可以在 [github](https://github.com/dylanwenzlau/ml-scripts/tree/master/meme_text_gen_convnet) 上查看所有代码，包括实用函数和一个样本训练数据文件。

# 第六步:？？？

# 第七步:显然，从人工智能生成的迷因中获利

结束了。