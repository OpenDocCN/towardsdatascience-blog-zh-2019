# 快速文本分类器是如何工作的？

> 原文：<https://towardsdatascience.com/fasttext-bag-of-tricks-for-efficient-text-classification-513ba9e302e7?source=collection_archive---------5----------------------->

当线性模型优于复杂的深度学习模型时

由*脸书研究院*开发的 FastText ，是一个高效学习单词表示和文本分类的库。FastText 支持单词和句子的监督(分类)和非监督(嵌入)表示。

然而，FastText 包的文档没有提供关于实现的分类器和处理步骤的细节。

在这里，我们试图跟踪 FastText 包的底层算法实现。

***简单地说，没有什么魔法，但是很少有聪明的步骤:***

*   通过平均单词/n-gram 嵌入来获得句子/文档向量。
*   对于分类任务，使用[多项式逻辑回归](https://en.wikipedia.org/wiki/Multinomial_logistic_regression)，其中句子/文档向量对应于特征。
*   当在具有大量类的问题上应用 FastText 时，可以使用分层的 softmax 来加速计算。

***【fast Text】高效文本分类锦囊妙计*** *:*

论文:[http://aclweb.org/anthology/E17-2068](http://aclweb.org/anthology/E17-2068)

*   从平均为文本表示的单词表示开始，并将它们馈送到线性分类器(多项式逻辑回归)。
*   作为隐藏状态的文本表示，可在要素和类之间共享。
*   Softmax 层，以获得预定义类别的概率分布。
*   Hierarchial Softmax:基于霍夫曼编码树，用于降低计算复杂度 *O(kh)* 到 O(hlog(k))，其中 k 是类的数量，h 是文本表示的维数。
*   使用一袋 n-grams 来保持效率而不损失准确性。没有明确使用词序。
*   使用哈希技巧来保持 n 元语法的快速和内存高效映射。
*   它是用 C++编写的，在训练时支持多重处理。

***FastText vs 深度学习进行文本分类:***

对于不同的文本分类任务，FastText 显示的结果在准确性方面与深度学习模型不相上下，尽管在性能上快了一个数量级。

*注*:简单的 FastText 算法的高精度表明，文本分类问题仍然没有被很好地理解，不足以构建真正有效的非线性分类模型。[【3】](https://pdfs.semanticscholar.org/9d69/93f60539d30ee325138b3465aa020fa3bcb4.pdf)

## **参考文献:**

[](https://github.com/facebookresearch/fastText) [## Facebook 研究/快速文本

### 快速文本表示和分类库。-Facebook 研究/快速文本

github.com](https://github.com/facebookresearch/fastText) [](/fasttext-under-the-hood-11efc57b2b3) [## 快速文本:引擎盖下

### 在这里我们将看到一个性能最好的嵌入库是如何实现的。

towardsdatascience.com](/fasttext-under-the-hood-11efc57b2b3) 

[https://pdfs . semantic scholar . org/9d 69/93f 60539d 30 ee 325138 b 3465 aa 020 fa 3 BCB 4 . pdf](https://pdfs.semanticscholar.org/9d69/93f60539d30ee325138b3465aa020fa3bcb4.pdf)

[https://gist . github . com/shagunsodhani/432746 f 15889 F7 F4 a 798 BF 7 f 9 EC 4 b 7d 8](https://gist.github.com/shagunsodhani/432746f15889f7f4a798bf7f9ec4b7d8)