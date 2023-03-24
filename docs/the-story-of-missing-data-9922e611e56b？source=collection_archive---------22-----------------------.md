# 缺失数据的故事

> 原文：<https://towardsdatascience.com/the-story-of-missing-data-9922e611e56b?source=collection_archive---------22----------------------->

## 机器学习

## 快速解释缺失数据在机器学习中的重要性以及如何处理它

![](img/0789364de362d14253b146db729dc85e.png)

Photo by [David Kennedy](https://unsplash.com/@dlewiskennedy?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/story?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

顾名思义，缺失数据意味着—

> 手边的数据集中缺少特征值或目标值

# 基于缺失原因的缺失数据分类

![](img/92cdf8f697ef1f3d9f6e1ecdb559e029.png)

**Randomness everywhere..** (Photo by [Jack Hamilton](https://unsplash.com/@jacc?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/shuffle?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText))

缺失数据基本上分为 3 类—

1.  **随机缺失(MAR)** —因为特征(预测值)或目标缺失随机值的原因可以从探索性数据分析本身来理解。(此处阅读关于此事的深入
2.  **完全随机失踪(MCAR)** —没有什么猫腻，失踪绝对是随机的！(在此阅读关于此的深入[)](https://medium.com/@danberdov/types-of-missing-data-902120fa4248)
3.  **不是随机遗漏(MNAR)**-对于机器学习爱好者或学生来说，这是最不幸的情况，因为这里遗漏的原因可以解释，但不能用数据集中的变量来解释。这种情况要求对数据分析的初始阶段(数据收集)进行彻底的调查！(在此阅读关于此的深入[)](https://medium.com/@danberdov/types-of-missing-data-902120fa4248)

# 基于变量的缺失数据分类

![](img/0b214d518c7824b16eeb1cbb76a6768b.png)

**Let the variables decide..**(Photo by [Nick Hillier](https://unsplash.com/@nhillier?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/numbers?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText))

在机器学习术语中，变量基本上有两种类型—

1.  特征/预测值和
2.  目标

因此，可以在这两个变量中找到缺失数据，因此可以分类如下:

*a.* 特征变量中缺失数据或**特征缺失**

*b.* 目标变量中的数据缺失或**目标缺失**——当目标变量本质上是离散的时，这种缺失也被称为**类缺失**，就像在二元或多类分类问题中一样。在连续目标变量的情况下，术语**目标缺失**可以用作默认值。

# 丢失数据的后果以及如何处理

![](img/1f24e3c944391217690241dba7e4d932.png)

**The challenges** (Photo by [Danny](https://unsplash.com/@findthevision?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/library?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText))

缺失数据提出了两个主要问题—

1.  **信息丢失** —这个问题与**特征缺失**有关，最好的处理方法是使用**插补技术**。一些插补技术有:均值、中值、众数或常值插补。
2.  **不平衡数据集** —此问题仅与类缺失相关，与目标缺失情况无关。这是大多数分类问题中的基本问题之一。这个问题的解决方案是使用像**过采样或欠采样技术**这样的技术。如果存在目标缺失问题，我的建议是删除对应于每个目标缺失案例的数据点。

# 丢弃还是估算？

![](img/0a9b6531d7b801e09d1c7851c1e04e6d.png)

**Which way?** (Photo by [Adi Goldstein](https://unsplash.com/@adigold1?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/which-way?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText))

以下是我自己的策略，只应如此重视:

> 如果**特征缺失案例的缺失值> 5%——丢弃特征本身，否则对一个或多个特征进行插补。**

5%或 10%或任何上限——由你决定。

> 基本思想是过多的插补是对基本分布行为的改变

# 过采样还是欠采样？

在解释过采样和欠采样之前，我们必须了解**多数阶级**和**少数阶级**。

## 多数阶级和少数阶级

![](img/0718fb6dfa593f3b4ee7b87a3d3b782e.png)

**Classes in society** ( Photo by [Jacek Dylag](https://unsplash.com/@dylu?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/crowd?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText))

这个术语与前面提到的分类问题密切相关。多数类是支配目标(或类)变量分布的离散变量值，份额最小的类是少数类。例如，在一项 100 人随机调查中，70 名男性代表多数群体，30 名女性代表少数群体。

> **过采样人为地将对应于少数类的新的或克隆的数据点引入数据集**并增加数据集的维数*，同时* ***将数据集从不平衡转换为完全平衡*** 。这意味着，过采样是针对少数阶级的。
> 
> **欠采样与过采样**相反——它消除了一些多数类数据点**以确保数据集完全平衡**。

## 一些过采样技术—

1.  引导或重采样(替换采样)-包括过采样和欠采样
2.  合成少数过采样技术( [SMOTE](https://medium.com/coinmonks/smote-and-adasyn-handling-imbalanced-data-set-34f5223e167) )和
3.  自适应合成采样方法

## 一些欠采样技术—

1.  托梅克-林克斯欠采样
2.  群集质心

## 混合技术——过采样+欠采样

1.  SMOTE 后跟 Tomek-Links

> 查看关于不平衡数据集重采样技术的简单 Kaggle 笔记本— [**此处**](https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets)

所有的过采样和欠采样技术 API 都可以在 sci-kit learn 的[“不平衡学习”](https://github.com/scikit-learn-contrib/imbalanced-learn)中找到

# 参考

1.  medium—[https://medium . com/@ danberdov/types-of-missing-data-902120 fa 4248](https://medium.com/@danberdov/types-of-missing-data-902120fa4248)
2.  medium—[https://medium . com/coin monks/smote-and-adasyn-handling-unbalanced-data-set-34f 5223 e167](https://medium.com/coinmonks/smote-and-adasyn-handling-imbalanced-data-set-34f5223e167)
3.  ka ggle—[https://www . ka ggle . com/RAF jaa/重采样-不平衡数据集策略](https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets)
4.  sci-kit learn @ Github—[https://github.com/scikit-learn-contrib/imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn)