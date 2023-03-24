# 评论评级预测:一种综合方法

> 原文：<https://towardsdatascience.com/review-rating-prediction-a-combined-approach-538c617c495c?source=collection_archive---------7----------------------->

## 结合评论文本内容和用户相似度矩阵来获取更多信息并改进评论评分预测

![](img/d6603914b19af42185617a50b8a3b472.png)

Source: [pixabay](https://pixabay.com)

# 开始

电子商务的兴起使顾客评论的重要性大大提高。网上有数百个评论网站，每种产品都有大量的评论。顾客已经改变了他们的购物方式，根据最近的 [**调查**](https://www.reviewtrackers.com/online-reviews-survey/) ，70%的顾客说他们使用评级过滤器过滤掉搜索中评级较低的商品。

对于支持这些评论的公司，如谷歌、亚马逊和 Yelp，成功判断评论是否对其他客户有帮助，从而提高产品曝光率的能力至关重要。。

有两种主要的方法来解决这个问题。第一种是基于评论文本内容分析并使用自然语言处理的原则(NLP 方法)。这种方法缺乏从顾客和商品之间的关系中得出的洞察力。第二个是基于推荐系统，特别是基于协同过滤，并且关注评论者的观点。使用用户的相似性矩阵和应用邻居分析都是该方法的一部分。此方法会忽略来自审查文本内容分析的任何信息。

为了获得更多的信息并提高评论评级的预测，研究人员在 [**这篇文章**](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4735905/) 中提出了一个结合评论文本内容和先前用户相似度矩阵分析的框架。然后，他们在两个电影评论数据集上做了一些实验，以检验他们的假设的有效性。他们得到的结果表明，他们的框架确实提高了评论评级的预测。这篇文章将描述我试图通过亚马逊评论数据集中的例子来跟踪他们的研究工作。记录这项工作的笔记本可以在这里[](https://github.com/Yereya/Amazon-baby-product-project)**获得，我鼓励在你的计算机上运行代码并报告结果。**

## ****数据****

**这里使用的数据集是由 UCSD 的 Julian McAuley 博士提供的。它包含亚马逊的产品评论和元数据，包括 1996 年 5 月至 2014 年 7 月期间的 1.428 亿条评论。产品评论数据集包含用户 ID、产品 ID、评级、有用性投票和每个评论的评论文本。
这里的数据可以找到**[](http://jmcauley.ucsd.edu/data/amazon/)**。******

## ********假设********

******在这项工作中，我的目标是检查研究人员的论文。它不是为这个问题找到最好的模型。我将试图证明，将以前已知的关于每个用户与其他用户相似性的数据与评论文本本身的情感分析相结合，将有助于我们改进用户评论将获得什么评级的模型预测。******

******![](img/145dc154f981b99f00ec07d88acb3abd.png)******

******Source: [pixabay](https://pixabay.com)******

## ********工作流程********

******首先，我将根据 RTC 分析执行 RRP。下一步将应用邻居分析来基于用户之间的相似性执行 RRP。最后一步将比较三种方法(基于 RTC 的 RRP、基于邻居分析的 RRP 以及两者的组合)并检查假设。******

## ********预处理********

******预处理在任何分析中都是一个关键步骤，在这个项目中也是如此。
主表的表头如下:******

******![](img/c20e7a2288787b3161e54dc62654c172.png)******

******The head of the primary table******

******首先，我删除了没有评论文本的行、重复的行和我不会用到的额外的列。
第二步是创建一个列，其中包含有用分子和有用分母相除的结果，然后将这些值分割到各个箱中。它看起来像这样:******

```
****reviews_df = reviews_df[~pd.isnull(reviews_df['reviewText'])]
reviews_df.drop_duplicates(subset=['reviewerID', 'asin', 'unixReviewTime'], inplace=**True**)
reviews_df.drop('Unnamed: 0', axis=1, inplace=**True**)
reviews_df.reset_index(inplace=**True**)

reviews_df['helpful_numerator'] = reviews_df['helpful'].apply(**lambda** x: eval(x)[0])
reviews_df['helpful_denominator'] = reviews_df['helpful'].apply(**lambda** x: eval(x)[1])
reviews_df['helpful%'] = np.where(reviews_df['helpful_denominator'] > 0,
                                  reviews_df['helpful_numerator'] / reviews_df['helpful_denominator'], -1)

reviews_df['helpfulness_range'] = pd.cut(x=reviews_df['helpful%'], bins=[-1, 0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                         labels=['empty', '1', '2', '3', '4', '5'], include_lowest=**True**)****
```

******最后一步是创建一个文本处理器，从杂乱的评论文本中提取有意义的单词。******

```
****def text_process(reviewText):
    nopunc = [i for i in reviewText if i not in string.punctuation]
    nopunc = nopunc.lower()
    nopunc_text = ''.join(nopunc)
    return [i for i in nopunc_text.split() if i not in stopwords.words('english')]****
```

******在被应用之后，这个 had -
1。删除了标点符号
2。转换成小写
3。移除了停用字词(在训练模型的上下文中不相关的字词)******

## ********一看数据********

******经过所有预处理后，主表的表头如下所示:******

******![](img/5e8d88e3d2156b24c61ed801f1d74e71.png)******

******下图显示了用户乐于助人的范围在产品评级中的分布情况:******

******![](img/015f5f9d5288c40f5510ecf6e260b905.png)******

******Heatmap******

******![](img/fd9bd181f15570c51a8644a5c196514a.png)******

******Barplot******

******人们很容易看出对高收视率的偏好。这种现象是众所周知的，这也在上面的 [**同一调查**](https://www.reviewtrackers.com/online-reviews-survey/) 中得到支持。根据这项调查:******

> ****“点评正越来越多地从消费者表达不满的地方，转变为在获得积极体验后推荐商品的地方”。****

****稍后，我将解释倾斜数据的问题是如何解决的(重采样方法)。****

# ******第一步:基于审核文本内容的 RRP******

## ******车型******

****为了检查和选择最佳模型，我构建了一个管道，它执行以下步骤。流水线将首先执行 TF-IDF 项加权和矢量化，然后运行分类算法。一般来说，TF-IDF 将使用我上面的“text_process”函数处理文本，然后将处理后的文本转换为计数向量。然后，它会应用一种计算方法，对更重要的单词赋予更高的权重。****

```
**pipeline = Pipeline([
    ('Tf-Idf', TfidfVectorizer(ngram_range=(1,2), analyzer=text_process)),
    ('classifier', MultinomialNB())
])
X = reviews_df['reviewText']
y = reviews_df['helpfulness_range']
review_train, review_test, label_train, label_test = train_test_split(X, y, test_size=0.5)
pipeline.fit(review_train, label_train)
pip_pred = pipeline.predict(review_test)
print(metrics.classification_report(label_test, pip_pred))**
```

****注意，我选择了 ngram_range = (1，2 ),算法是多项式朴素贝叶斯。这些决定是根据交叉验证测试的结果做出的。我所做的交叉验证测试超出了本文的范围，但是您可以在笔记本中找到它。
检查的车型有:
1。多项逻辑回归，作为基准
2。多项式朴素贝叶斯
3。决策树
4。随机森林****

****多项朴素贝叶斯给出了最好的准确度分数(0.61)，因此选择它做出的预测来表示基于 RTC 的 RRP。****

****这一步的最后一部分是将所选模型做出的预测导出到 csv 文件中:****

```
**rev_test_pred_NB_df = pd.DataFrame(data={'review_test': review_test2, 'prediction': pip_pred2})
rev_test_pred_NB_df.to_csv('rev_test_pred_NB_df.csv')**
```

# ******第二步:基于用户相似度的 RRP******

## ******预处理******

****在这一步中，用户相似性矩阵被构建，并且是我将计算每个用户之间的余弦相似性的基础。当我使用项目的名称构造矩阵时，出现了一些问题，但是通过转换为唯一的整数序列(与 SQL 中的 IDENTITY 属性相同)解决了这些问题。****

```
**temp_df = pd.DataFrame(np.unique(reviewers_rating_df['reviewerID']), columns=['unique_ID'])
temp_df['unique_asin'] = pd.Series(np.unique(reviewers_rating_df['asin']))
temp_df['unique_ID_int'] = range(20000, 35998)
temp_df['unique_asin_int'] = range(1, 15999)reviewers_rating_df = pd.merge(reviewers_rating_df, temp_df.drop(['unique_asin', 'unique_asin_int'], axis=1), left_on='reviewerID', right_on='unique_ID')reviewers_rating_df = pd.merge(reviewers_rating_df, temp_df.drop(['unique_ID', 'unique_ID_int'], axis=1),left_on='asin', right_on='unique_asin')reviewers_rating_df['overall_rating'] = reviewers_rating_df['overall']
id_asin_helpfulness_df = reviewers_rating_df[['reviewerID', 'unique_ID_int', 'helpfulness_range']].copy()# Delete the not in use columns:
reviewers_rating_df.drop(['asin', 'unique_asin', 'reviewerID', 'unique_ID', 'overall', 'helpfulness_range'], axis=1, inplace=True)**
```

****构建矩阵:为了节省处理时间，我使用 pivot 将数据转换成合适的形状，然后使用“csr_matrix”将其转换成稀疏矩阵。****

```
**matrix = reviewers_rating_df.pivot(index='unique_ID_int', columns='unique_asin_int', values='overall_rating')
matrix = matrix.fillna(0)
user_item_matrix = sparse.csr_matrix(matrix.values)**
```

## ******KNN 车型******

****我使用了 K-最近邻算法来进行邻居分析。KNN 模式易于实施和解释。相似性度量是余弦相似性，并且期望的邻居的数量是 10。****

```
**model_knn = neighbors.NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
model_knn.fit(user_item_matrix)**
```

****在训练阶段之后，我提取了邻居列表，并将其存储为一个 NumPy 数组。这产生了一个用户和与他们最相似的 10 个用户的二维数组。****

```
**neighbors = np.asarray(model_knn.kneighbors(user_item_matrix, return_distance=False))**
```

****下一步是获取 10 个最近的邻居，并将它们存储在数据帧中:****

```
**unique_id = []
k_neigh = []
for i in range(15998):
    unique_id.append(i + 20000)
    k_neigh.append(list(neighbors[i][1:10])) #Grabbing the ten closest neighborsneighbors_df = pd.DataFrame(data={'unique_ID_int': unique_id,
                                  'k_neigh': k_neigh})id_asin_helpfulness_df = pd.merge(id_asin_helpfulness_df, neighbors_df, on='unique_ID_int')
id_asin_helpfulness_df['neigh_based_helpf'] = id_asin_helpfulness_df['unique_ID_int']**
```

****最后，为了计算十个最接近的评论者写的评论的平均分，我编写了一个嵌套循环来遍历每一行。然后，循环将遍历用户的十个邻居，并计算他们的评论的平均得分。****

```
**for index, row in id_asin_helpfulness_df.iterrows():
    row = row['k_neigh']
    lista = []
    for i in row:
        p = id_asin_helpfulness_df.loc[i]['helpfulness_range']
        lista.append(p)
    id_asin_helpfulness_df.loc[index, 'neigh_based_helpf'] = np.nanmean(lista)**
```

# ******第三步:组合******

****![](img/87c1995ee94ca1b5330a66400981ea4e.png)****

****Photo by [ALAN DE LA CRUZ](https://unsplash.com/@alandelacruz4?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)****

****第三步，我导出了上面计算的结果，并将它们与所选模型的预测合并。然后，我有一个由四列组成的文件:
1)原始评论
2)他们得到的分数(地面真相)
3)第一步的预测(NLP 方法)
4)第二步的预测(用户相似性方法)
这两种方法的结合可以用许多不同的方式来完成。在本文中，我选择了简单的算术平均值，但其他方法也可以。除了上面的四列，我现在有了第五列:
5)列 3)和 4)中每一行的算术平均值****

# ******最后一步:报告******

****用于比较模型的度量是均方根误差(RMSE)。这是一个非常常见和良好的比较模型的措施。此外，我选择提出平均绝对误差(MAE ),因为它使用与测量数据相同的尺度，因此可以很容易地解释。结果如下所示:****

```
**RMSE for neigh_based_helpf: 1.0338002581383618
RMSE for NBprediction: 1.074619472976386
RMSE for the combination of the two methods: 0.9920521481819871
MAE for the combined prediction: 0.6618020568763793**
```

****组合方法的 RMSE 低于每种单独方法的 RMSE。****

# ******结论******

****总之，我的论文被证明是正确的。将关于每个用户与其他用户的相似性的先前已知数据与评论文本本身的情感分析相结合，确实有助于改进对用户评论将获得的评分的模型预测****

*****本文的目标是比较这些方法，看看研究人员提供的框架是否会提高预测的准确性。这不是为了找到基于 RTC 的最准确的 RRP 模型。*****

*****虽然 MAE 为 0.66 并不好，但这项工作的主要目的是检验假设，而不一定是寻求最佳的 RRP 模型。*****