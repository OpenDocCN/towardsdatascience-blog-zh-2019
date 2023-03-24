# 设备上的机器学习:Android 上的文本生成📝

> 原文：<https://towardsdatascience.com/on-device-machine-learning-text-generation-on-android-6ad940c00911?source=collection_archive---------10----------------------->

![](img/0c164a90cccd70031f00f219923d0019.png)

Photo by [Pereanu Sebastian](https://unsplash.com/@sebastian123?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

## 结合 GPT-2、TensorFlow 和 Kotlin 的力量，在移动设备上实现最先进的自然语言处理

在拥抱脸，我们的目标是解决和民主化自然语言处理(NLP)。目前，生产中的大多数模型都是在服务器上远程运行的，例如谷歌的搜索服务器。尽管如此，**移动设备上硬件的改进和对隐私的日益关注使得它们越来越适合离线运行**。

本文的目标是给出一个完全在设备上运行的用于文本生成的 Android 应用程序**的高级视图。代码在这里:[https://github . com/hugging face/TF lite-Android-transformers/tree/master/gp T2](https://github.com/huggingface/tflite-android-transformers/tree/master/gpt2)**

![](img/6cea666a19655cb32c71bb064cd2e0ef.png)

What we’re going to build 🤓

# 第一部分:将 GPT-2 转换为 TensorFlow Lite 格式

[GPT-2](https://openai.com/blog/better-language-models/) 是 2019 年发布的一款车型，其自然语言生成能力(NLG，NLP 的一个子集)令人印象深刻，以至于最大版本的发布被推迟了几个月。**你可以用** [**这个搞笑(吓人？)工具**](https://transformer.huggingface.co/doc/gpt2-large) **我们发布**。在这个应用程序中，我们将使用最小版本的模型。它的发电能力不如最大的那台令人印象深刻，但它的大小(500MB *对* 6GB)使它更适合移动使用！

在能够在设备上运行之前，**我们需要将它转换成合适的格式**、 [TensorFlow Lite](https://www.tensorflow.org/lite) (TFLite)。为此，我们可以运行以下 Python 脚本:

“[tf-nightly](https://pypi.org/project/tf-nightly/)” and “[transformers](https://pypi.org/project/transformers/)” libraries need to be installed in your environment. You can also try it directly in your browser using [this colab notebook](https://colab.research.google.com/drive/18JPzizAH995pd0iFWx4Xdf-sqjmZwHUD).

这个脚本使用了我们的[🤗Transformers](https://github.com/huggingface/transformers) 库导入“原始”模型，然后将其转换为 TFLite 格式。**注意脚本**的第 15/16 行:在运行转换之前，我们使用 TFLite 指定我们想要[将模型](https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-float16-quantization-halves-model-size-cc113c75a2fa)的权重(参数)量化为*半精度浮点格式*。这导致我们转换后的模型的最终大小为 237MB，即**原始“输入”模型大小的一半🎉。**不利方面？精确度损失极小，但考虑到节省的存储空间，在移动设备上绝对值得！

> 我们可以通过将权重转换为 8 位整数表示形式来进一步压缩我们的模型，结果只有 128MB。但是我们对这个版本的测试显示在设备上要慢得多。因此，我们更喜欢在这里使用半精度浮点版本。你仍然可以通过[改变默认模式](https://github.com/huggingface/tflite-android-transformers/tree/master/gpt2#change-the-model)来试验我们的应用程序的 8 位版本。

# 第二部分:将转换后的 GPT-2 模型集成到 Android 应用程序中

既然我们已经转换了我们的模型，我们可以专注于实际构建我们的应用程序。[GitHub](https://github.com/huggingface/tflite-android-transformers/tree/master/gpt2)上有完整的源代码，所以这里我只关注最有趣的部分。

在 Python 脚本中，我们指定(第 6/7 行)我们的模型将接受一个形状为 *[1，64]* 的**二维整数**数组作为输入，即类似这样的内容，其中内部数组包含 64 个元素:

```
[[142, 34, 100, 535, 30234, 45, 2934, ...]]
```

但是我们在现实生活中将要拥有的是一个字符串，对应于当前文本。因此，我们需要将该字符串转换成整数，*又称为* ***记号*** 。粗略地说，我们可以说,**一个令牌是我们的字符串**的一部分的数字表示。

*令牌*也是模型作为输出返回的内容。模型的每一次运行都允许我们确定文本的下一个标记，然后我们将它与前一个文本一起传递给我们的模型进行下一次运行，*等等* …

我们需要一些东西来把我们的字符串转换成记号，再把记号转换回字符串。这就是*记号赋予器*的作用。记号赋予器的两个主要功能通常是*编码*和*解码。*

Full implementation of the Tokenizer available [here](https://github.com/huggingface/tflite-android-transformers/blob/master/gpt2/src/main/java/co/huggingface/android_transformers/gpt2/tokenization/GPT2Tokenizer.kt)

*encode* 函数将我们的起始/前一个文本作为参数，使用正则表达式对其进行解析，然后将每个字符转换为特定的表示。它最后应用一个[字节对编码](https://en.wikipedia.org/wiki/Byte_pair_encoding) (BPE)算法，由于有了[模型词汇表](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json)，该算法的输出被映射成整数。🤯

*decode* 函数执行相反的操作，将标记映射到它们的词汇表示，然后将这个表示解码为最终的字符串。

现在我们知道如何编码和解码我们的文本，我们可以调用我们的模型！这是下面的*生成*功能的作用:

[Click here](https://github.com/huggingface/tflite-android-transformers/blob/master/gpt2/src/main/java/co/huggingface/android_transformers/gpt2/ml/GPT2Client.kt) to see the full implementation

该函数的输入是初始文本和我们想要生成的令牌数(*，即我们的模型被调用的次数*)。第一步是对我们的文本进行标记。

还记得我们说过模型的输入是一个形状数组*【1，64】*吗？我们需要去掉之前的文本标记，只保留最后的最大值。那是我们的*输入。*表示**下一个令牌的生成只依赖于这 64 个之前的令牌，忽略任何之前的令牌**。

> 当我们转换我们的模型时，我们可以指定一个更长的序列长度，但是这将意味着更多的推理计算，减慢我们的应用程序。这是我们这一代人在速度和“质量”之间的权衡。*🤔*

我们还需要创建数据结构，我们的模型将使用它的输出。我们的模型有许多输出，但是我们只对第一个“预测”感兴趣。

```
val predictions = Array(1) **{** Array(SEQUENCE_LENGTH) **{** FloatArray(VOCAB_SIZE) **} }**
```

当谈到多维数组时，我们在 Kotlin 的表达能力方面达到了一个极限；下面是它在 Java 中的样子:

```
float[][][] predictions = new float[1][SEQUENCE_LENGTH][VOCAB_SIZE]
```

我远不是一个 Java 迷，但是右边的表达对我来说似乎更容易读懂！

我们终于可以——了！—通过调用 TFLite 解释器来运行我们的模型:

```
tflite.runForMultipleInputsOutputs(arrayOf(inputIds), outputs)
```

一旦解释器填充了我们的“预测”数组，**我们需要确定将是我们的“下一个”令牌**。这样做有许多不同的方法；这里我们首先使用 *Top-K* 过滤，选择 *k* 更高的预测。然后，我们应用一个 [*Softmax* 函数](https://www.wikiwand.com/en/Softmax_function)来获得这些值的概率分布，然后通过多项式采样最终选择“那一个”。

# 第三部分:借助 Kotlin 协程，以用户界面友好的方式与活动交互

现在是时候将我们的模型链接到应用程序的接口了！**在设备上运行 GPT-2 等模型，即使是量化版本，也需要计算资源**。如果我们做错了，我们可能会在模型运行时以界面冻结而告终，这**对用户不太友好**！😱

为了避免这种糟糕的结果，**我们将使用** [**协程**](https://kotlinlang.org/docs/reference/coroutines-overview.html) **，这是在 Kotlin** 中进行非阻塞编程的一种非常好的方式。这是我们(几乎)完整的 *GPT2Client* 类，它是从我们的主活动加载的 [*ViewModel*](https://developer.android.com/topic/libraries/architecture/viewmodel) :

For full implementation, [check here](https://github.com/huggingface/tflite-android-transformers/blob/master/gpt2/src/main/java/co/huggingface/android_transformers/gpt2/ml/GPT2Client.kt)

该类首先需要加载我们模型的所有资产，并初始化 TFLite 解释器。为了在不阻塞 UI 的情况下做到这一点，在 *init* 块**中，我们启动了一个新的协程**，这要归功于[*viewmodelscope . launch*](https://developer.android.com/topic/libraries/architecture/coroutines)。在这个协程中，我们通过调用 3 个" *load"* 方法来加载我们的模型资产。以下是*负载模型*的签名:

```
private suspend fun loadModel() = **withContext(Dispatchers.IO)** **{** // Load the model file and initialize the interpreter with it... **}**
```

这里重要的是带有上下文的*(调度程序。*IO)部分。我们说**我们想在一个不同于主线程**的线程上执行这个方法，这里使用一个为 I/O 操作设计的线程([更多细节见这里](https://kotlin.github.io/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/-coroutine-dispatcher/index.html))。

> 通过 *viewModelScope.launch* 创建协程的“妙处”在于它将自己的生命周期与 *ViewModel* 的生命周期联系在一起。它确保当*视图模型*被清除时，协程被取消！*🙌*

然后，当用户点击应用程序中的“触发自动完成”按钮时，执行 *launchAutocomplete* 方法，**创建另一个协程**，我们将从其中调用我们的模型。

```
fun launchAutocomplete() {
    autocompleteJob = **viewModelScope.launch** **{** initJob.**join**()
        autocompleteJob?.**cancelAndJoin**()
        _completion.value = ""
        generate(_prompt.value!!)
    **}** }
```

在这个协程中，我们首先确保资产的初始化( *initJob* )已经完成，然后对潜在的先前模型运行( *autocompleteJob* )，**做同样的事情，如果还在运行**，我们就取消它。然后我们可以调用我们的*生成*方法:

```
private suspend fun generate(text: String, nbTokens: Int = 100) = **withContext(Dispatchers.Default)** **{** val tokens = tokenizer.encode(text)
    repeat (nbTokens) **{** // Run the model...
        // ... tokens.add(nextToken)
        val decodedToken = tokenizer.decode(listOf(nextToken))
        _completion.postValue(_completion.value + decodedToken)

        **yield()**
    **}
}**
```

用于此方法的调度员不是*调度员。IO* 因为我们在这里不做任何 I/O 操作，而是一个更通用的 ***调度程序。默认*** 使用共享后台线程的公共池*。*

这个方法另一个有趣的部分是[***yield()***](https://kotlin.github.io/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/yield.html)方法调用在*结束时重复*块。**这就是允许该方法检查最终取消的原因。没有它，就不可能取消，我们必须等到整个一代结束后才能释放资源！☠️** 这里我们在每次令牌生成后检查取消。

> 检查取消的另一种方式是检查 [*isActive* 属性](https://kotlin.github.io/kotlinx.coroutines/kotlinx-coroutines-core/kotlinx.coroutines/is-active.html)的值

由于使用了 [LiveData 结构](https://developer.android.com/topic/libraries/architecture/livedata)(我们的*完成*属性)，完成的文本然后“自动地”显示在应用程序中。🧙‍♀️

就是这样！在拥抱脸，我们相信我们只是处于人工智能在设备上运行的时代的开始。随着一方面专用硬件和相关驱动程序和框架的不断发展，另一方面量化和提取等技术的不断发展，**我们智能手机的功能有望拥有光明的未来**，允许以更高效和更高性能的方式运行更复杂的模型。

如果你想要更多的 Android 例子，你可以检查整个库[。我们还发布了](https://github.com/huggingface/tflite-android-transformers)[**一个回购协议，其中包含 iOS**](https://github.com/huggingface/swift-coreml-transformers) 的模型和应用，利用了苹果特有的 [CoreML](https://developer.apple.com/documentation/coreml) 框架。如果您对更深入的最新 NLP 感兴趣，我们的 [**🤗变形金刚**](https://github.com/huggingface/transformers) 库来了！