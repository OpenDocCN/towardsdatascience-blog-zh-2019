# 用 Python 构建自己的量子电路(带彩色图表)

> 原文：<https://towardsdatascience.com/building-your-own-quantum-circuits-in-python-e9031b548fa7?source=collection_archive---------8----------------------->

## 充满代码和彩色图表的直观指南…

*对于量子计算的速成班，请查看之前的* [*文章*](/quantum-computing-with-colorful-diagrams-8f7861cfb6da) *。*

***TLDR:*** *这里我说的是****Qiskit****，一个开源的 Python 模块，用于构建量子电路，模拟对量子位的操作。我们将浏览示例代码，然后通过非常直观、丰富多彩的图表深入解释代码的作用。*

![](img/38dfc8fbad812bc5d0cbd44aa0b2e797.png)

在我以前的文章中，我提到可以建立模型来模拟宇宙中感兴趣的过程。为了建立这些模型，我们使用量子电路，为此我们需要一个模块来操纵和运行量子设备上的量子程序。

在本文中，我们将关注[Qiskit](https://qiskit.org)——一个 Python 模块，由于其简单易读的语法，它使量子原型化成为一项无压力的任务。Qiskit 是由 IBM Research 开发的，旨在使任何人都可以更容易地与量子计算机进行交互。

## 量子电路构造新论

在我以前的文章中，我谈到了哈达玛，受控非和泡利 X 门。哈达玛门接收一个量子位元，并以相等的机率输出一个位元成为 1 或 0。

![](img/4db392a53e680485536abcef1a4ff33d.png)

The Hadamard Gate computes a Qubit with an equal probability of becoming a 1 or 0 when measured.

受控非(CNOT)门接收 2 个量子位，并且仅当控制量子位是 *ket 1* 时，将一个量子位从 *ket 0* 的一个状态翻转到 *ket 1* 。否则，它保持不变。

![](img/d702653d877bb20fad1b0e98b312669c.png)

The CNOT Gate flips a Qubit only if the control Qubit is ***ket 1***

现在我们已经概括了门的作用，我们终于可以继续定义我们自己的电路，并模拟它来获得量子位和比特的最终状态！

***注:*** *本教程不涉及泡利 X 门和 Y 门。*

## 建造我们自己的量子电路

由于 Qiskit 的大量文档和资源，构建自己的量子电路很容易。在进入更复杂的内容之前，让我们从头开始浏览构建电路的基础知识。这里，我们将把电路代码分解成容易理解的部分。我们开始吧！

在写量子电路之前，我们需要安装 Qiskit。为此，您可以简单地通过 ***pip*** 安装包:

```
**$** pip install qiskit
```

要测试您的安装，打开 Python 控制台并输入`import qiskit as qk`。如果它运行没有任何错误，您就可以开始了！

***注:*** *如果你遇到任何问题，Qiskit GitHub* [*问题*](https://github.com/Qiskit/qiskit/issues) *页面有大多数初学者问题的解决方案。但是，如果你还有任何迫切的问题，请随时在下面留言或通过*[*Twitter*](https://twitter.com/rishabh16_)*或*[*LinkedIn*](https://www.linkedin.com/in/rishabhanand16/)*联系我。*

现在我们已经配置了 Qiskit，让我们来处理一些量子位和比特。为了实例化 Qiskit 中的量子位和比特，我们可以使用`QuantumRegister`和`ClassicalRegister`对象来创建它们。这里，我们每种类型创建 2 个:

寄存器保存着我们的电路所使用的量子位和比特的记录。有了量子位和比特，让我们来构建我们的电路:

让我们在电路中添加一些门来操纵和处理我们的量子位和比特，并最终测量最终的量子位:

让我们想象一下我们的电路，这样我们就对应用于我们的量子位的转换有了一个粗略的想法:

如果我们将这个电路打印到控制台，我们可以看到一个方便打印的电路图(感谢 IBM！).它应该是这样的:

![](img/6916911d2dc0d7396205d4dc2047ac35.png)

The terminal output should look something like the diagram on the left. The image on the right is a visualization of how the circuit is supposed to be constructed.

恭喜你。你刚刚建立了一个量子电路。现在我们已经准备好了，让我们模拟电路，并使用仿真器运行它。Qiskit 提供具有各种模拟器后端的组件，这些组件以不同的方式操纵量子位。这些成分是空气，伊格尼丝，土地和水。

为了模拟电路，让我们使用 Aer 的`QasmSimulator`来模拟真实的量子设备。要使用 Aer 的模拟器，我们可以简单地创建一个模拟作业，并在最后收集结果。Qasm 模拟器运行电路 1024 次(默认)。

“Qasm 模拟器”听起来像是电子游戏里的东西。如果这听起来很吓人，不要担心。它所做的就是在你的本地机器上创建一个经典的模拟来模仿真实量子机器中会发生的事情。正如 Qiskit 文档所称，它始终是一个 ***伪模拟*** 。

最后，我们可以看到，电路输出“00”作为位串的概率为 0.481 (492/1024)，而输出位串为“11”的概率为 0.519 (532/1024)。

这样，通过模拟我们的电路，我们可以将这个位串转换成十进制或十六进制的对应形式，并查看它们给出的值。

## 问真正的问题

做完这一切，你可能会问，这一切有什么意义？我们对如何构建电路以及各个元件的功能有所了解。他们是怎么走到一起的？他们都是怎么融入的？这些都是合理的问题。

![](img/aba2bb7c816fbce8ae41e810f2058c19.png)

量子计算处理概率。事件发生的可能性。为此，我们假设事件是完全随机的。量子电路使我们能够获取量子位，将它们叠加，测量它们并得到最终结果。

为了说明这种完全无偏差随机性的应用，让我们构建一个随机数生成器！这样我们就可以看到叠加的力量，以及最终的结果是一个完全随机的数字，没有任何片面性。

此外，不要在本质上是经典的本地模拟上运行它(不能反映量子计算机的惊人性能)，让我们连接我们的电路，并在离您最近的实际 IBM 量子计算机上运行它！

![](img/cd43fd0bd727656d74e35da5ebe81a54.png)

A simplified diagram of the processes involved in building a quantum circuit, running it on a Quantum Computer and getting back the simulated results

在我们执行任何量子魔法之前，我们需要连接到离您所在地区最近的 IBM 量子计算机。要做到这一点，请访问 IBM Q [网站](https://quantumexperience.ng.bluemix.net/qx/login)并登录，如果您是新用户，请创建一个帐户。

登录后，单击位于右上角的配置文件图标。进入 ***我的账户*** 页面，进入 ***高级*** 标签页。在那里，您将看到一个用于 API 令牌的区域。如果你没有，那就生成一个吧。

![](img/b0aace8d512ca50ff32b812a4d4deecf.png)

You should see a token inside. If not, click Regenerate to generate one. I’ve blocked mine for security reasons.

我们终于可以开始编写一个客户端电路，它与离您最近的 IBM Q 机器接口，如左边的面板所示。

![](img/55d394c44fcadfe4643cef6b97b80688.png)

The available machines closest to where I am

在本教程中，我将使用 ibmqx4 系统，它最多可以保存和操作 4 个量子位。根据您所在的位置，您可能会在仪表板上看到其他可用的机器，所以如果您看不到我的机器，请不要担心。选择你喜欢的那个(选择听起来最酷的那个😎).

接下来，让我们使用下面几行代码连接到量子计算机:

在我们构建电路之前，让我们为随机数生成器创建一个量子位和经典位的记录，该随机数生成器生成 0 到 7 (n=3)之间的数字。

现在，我们需要构建一个电路，返回一个量子位的叠加，并将其折叠为一个表示 0 和 2^n-1 之间的整数的位串。要做到这一点，我们可以在 3 个量子位上应用哈达玛门，让它们以 50%的概率叠加成 1 或 0。

如果我们想象这个电路，它会是这样的:

![](img/0ec3ab4cc785e4a3f69e571862bbd074.png)

The Quantum Circuit we’re sending up to the IBM Q Quantum Computer

现在，是激动人心的时候了。让我们实例化将要运行我们电路的量子后端。还记得你在上面使用的 Qasm 模拟器吗？同理，我的 ***ibmqx4*** 机器就是这里的模拟器(这里，是真货！).让我们连接我们的后端模拟器:

*Shots* 这里指的是电路运行的次数。这里我们只想要 1 个随机生成的数字，所以我们将其设置为 1。

***注意:*** *您可以随意使用这个数字，将其设置为 1000 左右，以查看 1000 次运行后生成的数字的直方图。*

接下来是主事件函数，它在 ***ibmqx4*** 量子计算机上运行一个作业，并返回一个 0 到 7 之间的完全随机数:

现在，是关键时刻了。让我们在分配给我们的量子计算机上运行完整的电路:

这里的代码将需要几分钟的时间来运行，并从我的帐户中扣除了 3 个积分(开始时给你 15 个)。当我运行电路时，最终输出是 **6** 。

就这样，祝贺你走了这么远！你刚刚建立了一个量子电路。让我们回顾一下成功运行量子程序的步骤:

![](img/29d0de64ec983fd65f7e24622f6b0366.png)

The typical process of building a Quantum Circuit and running it on a Quantum backend.

***注:*** *量子随机数发生器的完整代码可以在* [*这里*](https://gist.github.com/rish-16/6f34b7481abbe4257216cdd032cbe78b) *找到。*

## 包扎

你刚刚建立了一个量子电路！太棒了。你在成为专家的道路上前进了一步。通过思考需要处理比特的简单过程，并将问题转化为可以由量子计算机解决的问题，这是牢固掌握概念的好方法。

Congrats on building a Quantum Circuit! The man himself wishes you well!

## 简单地

学习量子计算背后的原理肯定是非常困难的！有这么多的理论在流传，对正在发生的事情有一个基础的理解是非常重要的。

学习 Qiskit 真的很有趣，IBM Research 提供的文档和资源对我理解正在发生的事情帮助很大。在包括 Google Cirq 和 QuTiP 在内的所有量子处理器中，Qiskit 是最容易学习和实现电路的。

量子计算正以充满希望的速度前进。研究人员正在普及量子计算机的使用，以执行任务和解决曾经被认为不可能解决的问题。

但是，量子计算理论还没有达到让所有人都能完全接触到的阶段。这可能是由于量子计算机的笨拙，庞大的设备和保持该领域活跃所需的大量资源。

![](img/123b989c885e2bdefaae9c1b279e6de9.png)

尽管如此，希望这篇文章能让你深入了解这个领域正在发生的事情，并激发你的兴趣！如果你有任何问题或想谈论什么，你可以在推特(Twitter)或领英(LinkedIn)上看到我公然拖延。

我的下一篇文章可能会关注量子计算背后的数学，以及不同的门和逻辑如何操纵量子位。我们将简要介绍量子算法以及它们是如何工作的！

在那之前，我们下一集再见！

[里沙卜·阿南德](https://medium.com/u/50c04ecf0ec5?source=post_page-----e9031b548fa7--------------------------------)的原创文章

## 相关文章

一定要看看我关于技术和机器学习的其他文章，以及对未来软件创新的评论。

[](/quantum-computing-with-colorful-diagrams-8f7861cfb6da) [## 量子计算速成班使用非常丰富多彩的图表

### 几乎所有你需要知道的关于量子计算的东西都用非常直观的图画来解释…

towardsdatascience.com](/quantum-computing-with-colorful-diagrams-8f7861cfb6da) [](https://medium.com/sigmoid/https-medium-com-rishabh-anand-on-the-origin-of-genetic-algorithms-fc927d2e11e0) [## 遗传算法综合指南(以及如何编码)

### 遗传算法的起源

medium.com](https://medium.com/sigmoid/https-medium-com-rishabh-anand-on-the-origin-of-genetic-algorithms-fc927d2e11e0) [](https://hackernoon.com/catgan-cat-face-generation-using-gans-f44663586d6b) [## CatGAN:使用 GANs 生成猫脸

### GANs 的详细回顾以及如何跟他们浪费时间…

hackernoon.com](https://hackernoon.com/catgan-cat-face-generation-using-gans-f44663586d6b) [](https://hackernoon.com/introducing-tensorflow-js-3f31d70f5904) [## TensorFlow.js 简介🎉

### 在浏览器上运行 ML 模型变得更加容易！

hackernoon.com](https://hackernoon.com/introducing-tensorflow-js-3f31d70f5904)