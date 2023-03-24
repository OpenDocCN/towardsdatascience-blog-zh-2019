# 提高机器学习代码性能的 4 个简单步骤

> 原文：<https://towardsdatascience.com/4-easy-steps-to-improve-your-machine-learning-code-performance-88a0b0eeffa8?source=collection_archive---------9----------------------->

![](img/9093627bca4ca729b054dde2760b726f.png)

## 我想向你展示 4 个简单的步骤，这将帮助你，在没有任何优化知识的情况下，至少将你的代码加速两次，并使它更有效和可读。

每天有多少新的机器学习库和模型被创建出来，这真的令人震惊。但令人失望的是，它们中的大部分都没有经过适当的设计。实际上，有时背后有好想法的好东西只是没有被使用，因为它没有被很好地记录，有困难或不直观的 API，或者只是因为它没有很好地执行并且对于生产环境来说太慢。一个很好的例子就是我在上一篇文章中提到的库— [normalize](https://github.com/EFord36/normalise) 。这是一个美丽的想法，它产生的结果确实可以在一些 nlp 任务中产生影响，但其中有一些意想不到的错误，API 不够灵活。好吧，你说，这两件事并不重要，实际上，你可以挺过去。但是你不能忍受的是它的性能很糟糕，它只是**慢**，所以不能在生产环境中使用。

我想鼓励每个人，那些创建机器学习库或只是展示自己的可重用代码片段的人，尽可能地优化它。因为每个人，或多或少都想有所作为或让这个世界变得更好一点，帮助其他人缩短他们走过的路，并立即获得更好的结果。我想向你展示 4 个简单的步骤，这将帮助你，在没有任何优化知识的情况下，至少将你的代码加速两次，并使它更有效和可读。

# 高效地计算你的数学

船长明显说:“使用你的工作与图书馆的数学函数或只是使用 numpy，你会一直没事”。他是对的，因为在大多数情况下，库的作者在某种程度上优化了所有的函数，使他的代码尽可能的快。但是几乎每个 python ML 库都是基于 numpy 的。Numpy [官方网页](https://www.numpy.org/)上写着:

> NumPy 是使用 Python 进行科学计算的基础包。除其他外，它包含:
> 
> 一个强大的 N 维数组对象
> 
> 复杂的(广播)功能
> 
> 集成 C/C++和 Fortran 代码的工具
> 
> 有用的线性代数、傅立叶变换和随机数功能

实际上，在 ML 中我们只是不断地做一些矩阵运算。Numpy 使用你能想到的所有可能的最先进的优化技术来提高所有这些操作的效率。所以规则一:

> 用 numpy 代替 python math，用 numpy 数组代替 python 数组。

如果你遵循这第一条简单的规则，你很快就会感觉到不同。

# 向量化您的函数

如何摆脱恼人的 for 循环，它只是在一组输入上应用相同的函数？把这个函数矢量化就行了！Numpy 允许您向量化一个函数，这意味着新创建的函数将应用于列表输入并产生一个函数结果数组。根据这篇[帖子](/data-science-with-python-turn-your-conditional-loops-to-numpy-vectors-9484ff9c622e)，它可能会让你的计算速度提高至少两倍。我强烈推荐阅读它，因为作者展示了 forloops 和 numpy 矢量化的漂亮对比，并展示了测量结果，证明 numpy 矢量化工作更快！让我们来看一些简单的代码片段，来展示它是如何工作的:

```
import numpy as np
from math import sin, cos, gammadef func(x):
    if x > 10 * sin(x) * cos(x):
        return sin(x) * cos(x)
    elif x**100 < (x-1)**101:
        return x**89
    elif x**100 - gamma(x) < (x-1)**101 * gamma(x):
        return gamma(x)
    else:
        return x**56arr = np.random.randn(10000)# FoorLoop method
res = []for i in arr:
    res.append(func(i))# Numpy Vectorizationfunc_vect = np.vectorize(func)
res_vet = func_vect(arr)
```

在这种情况下，Numpy 矢量化代码的运行速度提高了 2 倍！矢量化不仅有助于加快代码速度，还能使代码更加整洁。例如，使 nltk 中的词干处理在所有令牌上运行，而不是只在一个令牌上运行:

```
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenizetokens = word_tokenize(text)
porter = PorterStemmer()# vectorizing function to able to call on list of tokens
stem_words = np.vectorize(porter.stem)tokens_stems = stem_words(tokens)
```

它看起来比一个看起来更好的选择:

```
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenizetokens = word_tokenize(text)
porter = PorterStemmer()def stem_words(tokens):
    stems = []
    for t in tokens:
        stems.append(porter.stem(t))
    return stemstokens_stems = stem_words(tokens)
```

总之，矢量化是一种非常强大的机制，它不仅可以帮助您加快代码速度，还可以让代码看起来更整洁。

# sklearn 中的 n_jobs

现在几乎每台计算机都有不止一个内核。不使用所有的，如果你付了钱，对我来说是一种犯罪。事实上，你必须从你的电脑上拿走它能给你的一切。对于 ML 开发者来说，他的计算机不必是一个漂亮和花哨的东西，首先，它是一个勤奋的矿工，必须像苏联的 Stakhanov 一样分三班工作。你只需要从你的电脑上拿走它能给你的一切，如果你仍然不明白为什么，只要记住一件简单的事情:“你已经为此付出了代价”。ML 开发者在他的管道中使用更多内核的最简单方法就是不要忘记在他使用的每个 sklearn 模型中设置 **n_jobs** 参数。 **n_jobs** 告诉 sklearn 并行运行多少个作业来拟合和预测模型。大多数 sklearn 模型支持并行训练和转换。这将有助于您在大型或高维数据集上搜索最佳参数或训练模型时节省大量时间。注意不要将 **n_jobs** 设置为 **-1** ，因为并行化节省的是你的时间，而不是 RAM。是的，实际上并行化可能会占用大量内存，但是使用至少 2 个作业而不是 1 个作业已经可以帮助您优化模型训练和预测性能。

# MapReduce

维基百科说:

> **MapReduce** 是一种编程模型和相关实现，用于在集群上使用并行分布式算法处理和生成大数据集。

看起来很复杂，就像只能在大型集群或大型数据集上使用的东西，对吗？但是这很简单，我将用几句话和一个例子来解释。

MapReduce 分为两部分:

1.  map——函数，它以某种方式将输入修改为输出。这个函数在每个需要处理的数据样本上被调用。这一步是并行的，因为对于每个样本，事实上，你做同样的操作。
2.  Reduce 函数，聚合 map 函数的所有输出。这一步通常是连续的。

例如，您必须计算一组数据的平方和。顺序实施看起来像这样:

```
arr = [1, 2, 3, 4, 5]result = 0
for i in arr:
    result += i**2
```

让我们将这个任务转换成 MapReduce:

1.  地图—计算数字的平方
2.  减少-计算地图结果的总和

python 中的顺序实现:

```
from functools import reducearr = [1, 2, 3, 4, 5]map_arr = map(lambda x: x**2, arr)
result = reduce(lambda x, y: x + y, map_arr)
```

如前所述，map 部分将被并行化，reduce 将保持顺序。让我们看一下并行化实施:

```
import multiprocessing as mp
from functools import reducedef pow2(x):
    return x**2arr = [1, 2, 3, 4, 5]# count of avaliable cores, of course you can use less for computations
cores = mp.cpu_count()
pool = mp.Pool(cores)map_arr = pool.map(pow2, arr)
result = reduce(lambda x, y: x + y, map_arr)pool.close()
pool.join()
```

所以现在对于这个任务，你使用你的计算机的所有力量！**。map 池对象的**方法假设 map 函数只有一个参数，有一个等价的**。星图**、**、**其中假设了不止一个自变量的星图函数，这些函数的完整描述你可以在这里找到[。值得一提的是，两者都是**。映射**和**。starmap** 按照与输入参数相同的顺序返回映射值。](https://docs.python.org/3.7/library/multiprocessing.html)

有时，不仅在一个样本上，而且在一批样本上计算 map 函数更有效。如果计算任务足够简单和快速，多处理计算的主要开销将是在进程间传输数据，这会大大降低进程速度。在我的帖子的前一部分，我们讨论了 sklearn 模型的 **n_jobs** 参数，让我们创建 base transformer 的通用代码片段，它可以并行化其转换过程，在这种情况下，批处理将是最佳选择。

下面是 pandas 在基本 sklearn 转换器中应用并行化的代码片段，您可以在以后的管道中使用它:

```
import multiprocessing as mp
from sklearn.base import TransformerMixin, BaseEstimatorclass ParallelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 n_jobs=1):
        """
        n_jobs - parallel jobs to run
        """
        self.variety = variety
        self.user_abbrevs = user_abbrevs
        self.n_jobs = n_jobs def fit(self, X, y=None):
        return self def transform(self, X, *_):
        X_copy = X.copy() cores = mp.cpu_count()
        partitions = 1

        if self.n_jobs <= -1:
            partitions = cores
        elif self.n_jobs <= 0:
            partitions = 1
        else:
            partitions = min(self.n_jobs, cores)

        if partitions == 1:
            # transform sequentially
            return X_copy.apply(self._transform_one)

        # splitting data into batches
        data_split = np.array_split(X_copy, partitions)

        pool = mp.Pool(cores)

        # Here reduce function - concationation of transformed batches
        data = pd.concat(
            pool.map(self._preprocess_part, data_split)
        )

        pool.close()
        pool.join() return data def _transform_part(self, df_part):
        return df_part.apply(self._transform_one) def _transform_one(self, line): # some kind of transformations here
        return line
```

我来总结一下这段代码。确定是并行还是顺序处理数据集。如果是并行的，将所有数据分成 **n_jobs** 批并并行转换，然后连接回数据集的转换批。仅此而已！我希望这个代码片段能够帮助您提升您的转换管道。

我希望你喜欢我的帖子，如果你有进一步的问题，请在评论中提问，我很乐意回答。期待您的反馈！

附言:如果你是一个有经验的开发人员，想要为开源做贡献，你可能的任务之一就是帮助好的但未优化的库变得更快并准备好生产。如果不知道，从哪里入手，可以帮助[规格化](https://github.com/EFord36/normalise)库，优化其性能。我和这个库的作者没有任何联系，他们不付我钱，它也不是一个广告。只是另一个图书馆，需要你的帮助；)