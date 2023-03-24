# “嘿，那是什么？”使用解释调试预测

> 原文：<https://towardsdatascience.com/https-towardsdatascience-com-that-debugging-predictions-using-explanations-861cc94cb875?source=collection_archive---------40----------------------->

机器学习(ML)模型到处都在涌现。有许多技术创新(例如，深度学习、可解释的人工智能)使它们更加准确，适用范围更广，可供更多人在更多商业应用中使用。名单无处不在:[银行业](https://www.techradar.com/news/how-ai-and-machine-learning-our-improving-the-banking-experience)，[医疗保健](https://www.accenture.com/us-en/insight-artificial-intelligence-healthcare)，[科技](https://www.wired.com/story/guide-self-driving-cars/)，[以上](https://www.forbes.com/sites/forbestechcouncil/2018/09/27/15-business-applications-for-artificial-intelligence-and-machine-learning/#413d9992579f)全部。

然而，与任何计算机程序一样，模型也会有错误，或者更通俗地说，就是[bug](https://web.archive.org/web/20000818180803/http://www.jamesshuggins.com/h/tek1/first_computer_bug_large.htm)。发现这些 bug 的过程与以前的技术有很大不同，[需要一个新的开发者栈](https://blog.fiddler.ai/2019/06/ai-needs-a-new-developer-stack/)。“[很快我们就不会给电脑编程了，我们会像训练狗一样训练它们](https://www.wired.com/2016/05/the-end-of-code/)”(《连线》，2016)。“[梯度下降能比你写的代码更好。对不起](https://medium.com/@karpathy/software-2-0-a64152b37c35)”(安德烈·卡帕西，2017)。

在深度学习神经网络中，我们看到的可能是数百万个连接在一起的权重，而不是人们编写的一行行代码，形成了一个不可理解的网络。([图片鸣谢](https://commons.wikimedia.org/wiki/File:Two-layer_feedforward_artificial_neural_network.png))

![](img/8caf7e9bba0bcba75d4785eec6e526fc.png)

How do we find bugs in this network?

那么我们如何在这个网络中找到 bug 呢？一种方法是解释你的模型预测。让我们看看通过解释可以发现的两种类型的错误(数据泄漏和数据偏差)，并用预测贷款违约的例子进行说明。这两个其实都是数据 bug，但是一个模型总结了数据，所以在模型中表现出来。

大部分 ML 模型都是有监督的。您[选择](https://blog.fiddler.ai/2019/05/humans-choose-ai-does-not/)一个精确的预测目标(也称为“预测目标”)，收集一个具有特征的数据集，并用目标标记每个示例。然后你训练一个模型使用这些特征来预测目标。令人惊讶的是，数据集中经常存在与预测目标相关但对预测无用的要素。例如，它们可能是从未来添加的(即，在预测时间之后很久)，或者在预测时间不可用。

这里有一个来自 [Lending Club 数据集](https://github.com/fiddler-labs/p2p-lending-data)的例子。我们可以使用这个数据集尝试用 loan_status 字段作为我们的预测目标来建模预测贷款违约。它取值为“全额支付”(好的)或“注销”(银行宣布损失，即借款人违约)。在这个数据集中，还有 total_pymnt(收到的付款)和 loan_amnt(借入的金额)等字段。以下是一些示例值:

![](img/91cdd1ecd2f804b7642ba8bbb9f3f83c.png)

*Whenever the loan is “Charged Off”, delta is positive. But, we don’t know delta at loan time.*

注意到什么了吗？每当贷款违约(“冲销”)时，总支付额小于贷款额，delta (=loan_amnt-total_pymnt)为正。这并不奇怪。相反，这几乎是违约的定义:在贷款期限结束时，借款人支付的金额少于贷款金额。现在，德尔塔没有*有*肯定违约:你可以在偿还全部贷款本金而不是全部利息后违约。但是，在这个数据中，98%的情况下，如果 delta 为负，则贷款已全部付清；并且 100%的时间 delta 是正的，贷款被注销。包含 total_pymnt 给了我们近乎完美的信息，但是我们直到整个贷款期限(3 年)结束后才得到 total_pymnt！

在数据中包含 loan_amnt 和 total_pymnt 有可能实现近乎完美的预测，但是我们不会真正使用 total_pymnt 来完成真正的预测任务。将它们都包含在训练数据中是预测目标的*数据泄漏*。

如果我们做一个(作弊)模型，它会表现得很好。太好了。而且，如果我们对一些预测运行特征重要性算法(模型解释的一种常见形式)，我们将看到这两个变量变得很重要，如果运气好的话，会发现这种数据泄漏。

下面，Fiddler 解释界面显示“delta”是提高这个示例预测的一个重要因素。

![](img/53f066c97e40678127b1a9f2ca0e62cf.png)

“delta” really stands out, because it’s data leakage.

这个数据集中还有其他更微妙的潜在数据泄漏。例如，等级和子等级由 Lending Club 专有模型分配，该模型几乎完全决定利率。所以，如果你想在没有 Lending Club 的情况下建立自己的风险评分模型，那么 grade、sub_grade、int_rate 都是数据泄露。他们不会让你完美地预测违约，但想必他们会帮忙，否则 Lending Club 不会使用他们自己的模型。此外，对于他们的模型，他们包括 [FICO 评分](https://www.fico.com/en/products/fico-score)，另一个专有的风险评分，但大多数金融机构购买和使用。如果你不想用 FICO score，那么那也是数据泄露。

数据泄漏是任何你不能或不愿用于预测的预测数据。建立在有泄漏的数据上的模型是有缺陷的。

假设由于糟糕的数据收集或预处理中的错误，我们的数据有偏差。更具体地说，在特征和预测目标之间存在虚假的相关性。在这种情况下，解释预测将显示一个意想不到的特点往往是重要的。

我们可以通过删除邮政编码从 1 到 5 的所有注销贷款来模拟贷款数据中的数据处理错误。在这个 bug 之前，邮政编码并不能很好地预测 chargeoff(一个 0.54 的 [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) ，仅略高于 random)。在这个 bug 之后，任何以 1 到 5 开头的邮政编码都不会被注销，AUC 会跳到 0.78。因此，在根据邮政编码的数据预测(无)贷款违约时，邮政编码将成为一个重要特征。在本例中，我们可以通过查看邮政编码很重要的预测来进行调查。如果我们善于观察，我们可能会注意到这种模式，并意识到这种偏见。

如果用邮政编码的第一个数字来概括，下面是销账率的样子。一些邮政编码没有冲销，而其余的邮政编码的比率与整个数据集相似。

![](img/7e238c71b59de8e3fe77768fe57bc240.png)

In this buggy dataset, there are no charged-off loans with zip codes starting with 6, 7, 8, 9, 0.

下面，Fiddler 解释 UI 显示邮政编码前缀是降低这个示例预测的一个重要因素。

![](img/2b18fda64f929e2c6aae3b5e152932cd.png)

“zip_code_prefix” really stands out, because the model has a bug related to zip code.

从这种有偏差的数据中建立的模型对于根据我们还没有看到的(无偏差的)数据进行预测是没有用的。它只在有偏差的数据中是准确的。因此，建立在有偏差数据上的模型是有缺陷的。

对于不涉及模型解释的模型调试，还有许多其他的可能性。例如:

1.  查找[过度配合](https://en.wikipedia.org/wiki/Overfitting)或配合不足。如果你的模型架构过于简单，它将会不适应。如果太复杂，就会过拟合。
2.  对你理解的黄金预测集进行回归测试。如果这些失败了，您也许能够缩小哪些场景被破坏。

由于解释不涉及这些方法，我在这里就不多说了。

如果您不确定模型是否正确使用了数据，请使用要素重要性解释来检查其行为。您可能会看到数据泄漏或数据偏差。然后，您可以修复您的数据，这是修复您的模型的最佳方式。

Fiddler 正在构建一个[可解释的人工智能引擎](https://fiddler.ai/)，它可以帮助调试模型。请发邮件至 [info@fiddler.ai](mailto:info@fiddler.ai) 联系我们。

*原载于 2019 年 7 月 22 日*[*https://blog . fiddler . ai*](https://blog.fiddler.ai/2019/07/debugging-predictions-using-explanations/?preview=true)*。*