# 用于机器学习的 SciKit 库中的 Fit 与 Transform

> 原文：<https://towardsdatascience.com/fit-vs-transform-in-scikit-libraries-for-machine-learning-3c70e6300ded?source=collection_archive---------11----------------------->

![](img/e562567e8ef8adc2089274094adfeba4.png)

我们已经在很多 SciKit 的库中看到了 fit()、transform()和 fit_transform()这样的方法。而且几乎所有的教程，包括我写过的，都只告诉你只用其中一种方法。这里出现的一个明显的问题是，这些方法是什么意思？你说的适合某样东西，改造某样东西是什么意思？transform()方法有一定的意义，它只是转换数据，但是 fit()呢？在这篇文章中，我们将试着理解两者之间的区别。

为了更好地理解这些方法的含义，我们将以 import 类为例，因为 import 类有这些方法。但是在我们开始之前，请记住，拟合像估算器这样的东西不同于拟合整个模型。

您使用一个估算器来处理数据集中的缺失数据。Imputer 为您提供了用列的平均值或甚至中值替换 NaNs 和空格的简单方法。但是在替换这些值之前，它必须计算将用于替换空白的值。如果您告诉估算者，您希望用该列中所有值的平均值替换该列中的所有变量，估算者必须首先计算平均值。计算该值的这一步称为 fit()方法。

接下来，transform()方法将使用新计算的值替换列中的 nan，并返回新的数据集。这很简单。fit_transform()方法将在内部完成这两项工作，并且通过只公开一个方法来简化我们的工作。但是有些情况下，您只想调用 fit()方法和 transform()方法。

在训练模型时，您将使用训练数据集。在该数据集上，您将使用估算器，计算值，并替换空白。但是，当您在测试数据集上拟合这个训练好的模型时，您不需要再次计算平均值或中值。您将使用与训练数据集相同的值。为此，您将对训练数据集使用 fit()方法，仅计算该值并将其保存在 input 中。然后，您将使用相同的 Inputer 对象对测试数据集调用 transform()方法。这样，为训练集计算的值(保存在对象内部)也将用于测试数据集。

简单地说，您可以对训练集使用 fit_transform()方法，因为您需要拟合和转换数据，并且您可以对训练数据集使用 fit()方法来获取值，然后用它来转换()测试数据。如果你有任何意见或不能理解，请告诉我。

> 在 [Twitter](https://twitter.com/contactsunny) 上关注我，了解更多[数据科学](https://blog.contactsunny.com/tag/data-science)、[机器学习](https://blog.contactsunny.com/tag/machine-learning)，以及通用[技术更新](https://blog.contactsunny.com/category/tech)。还有，你可以[关注我的个人博客](https://blog.contactsunny.com/)。