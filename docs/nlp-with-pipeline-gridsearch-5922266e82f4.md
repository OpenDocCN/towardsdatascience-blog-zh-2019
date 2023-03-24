# 具有管道和网格搜索的 NLP

> 原文：<https://towardsdatascience.com/nlp-with-pipeline-gridsearch-5922266e82f4?source=collection_archive---------16----------------------->

![](img/c3a0521ad4d29eab4b672ebb0c1af57b.png)

[Source](http://Photo by Rodion Kutsaev on Unsplash)

自然语言处理，简称 NLP，在 Alexa、Siri 和谷歌助手等人工智能助手中风靡一时。当我第一次在钢铁侠电影中看到 J.A.R.V.I.S .的时候，好像就在昨天。看到这一点激发了我对我们如何与计算机互动的兴趣。一台能像人一样听和反应的电脑？它是如何工作的？这看起来像魔术，但实际上计算机正在将单词分解成更简单的形式，在某些情况下，还会创建这些形式的列表进行排序。计算机寻找特定单词之间的模式或关系，然后根据这些关系做出预测。

我们将使用 Pipeline 和 GridSearch 来运行几个模型，并确认哪一个模型最适合预测给定博客主题的帖子的位置。例如，如果我的帖子会出现在 NHL 或 Fantasy Hockey 中，一个模型能够有效地判断出来吗？为此，我将使用我自己的数据集，但是代码在大多数情况下是可重复的。

![](img/420e9a995ce8fa6a9e7b58a445a688a3.png)

[Source](http://Photo by Andrik Langfield on Unsplash)

# 让我们开始建设吧！

首先，您需要导入库和数据集，这一步我不会显示导入，因为列表会变得很长。我们将在数据已经被清理并且处于可以开始建模的形式之后开始。

让我们声明我们的变量！我们将需要创建 X 和 Y 变量，并使用训练测试分割。

```
# Creating X, y Variables
X, y = df[‘post’], df[‘blog_topic’]# Setting up train test split
X_train, X_test, y_train, y_test = train_test_split(X, y)
```

我们在这里所做的是将我们的数据集分成两个数据集:训练数据和测试数据。现在，我们已经将 X 变量声明为博客文章，将 Y 变量声明为博客主题。我们希望我们的模型接受一个博客帖子，并预测该帖子是否应该在 NHL 或 Fantasy Hockey 主题中。

# 管道

Pipeline 将接收带有某些参数的模型，然后我们可以通过 GridSearch 来查看哪个模型和参数给出了最好的结果。

```
# Pipeline & Gridsearch setup
# TFIDF pipeline setup
tvc_pipe = Pipeline([
 (‘tvec’, TfidfVectorizer()),
 (‘mb’, MultinomialNB())
])# Randomforest pipeline setup
rf_pipe = Pipeline([
 (‘tvec’, TfidfVectorizer()),
 (‘rf’, RandomForestClassifier())
])# Fit
tvc_pipe.fit(X_train, y_train)
rf_pipe.fit(X_train, y_train)# Setting params for TFIDF Vectorizer gridsearch
tf_params = {
 ‘tvec__max_features’:[100, 2000],
 ‘tvec__ngram_range’: [(1, 1), (1, 2), (2, 2)],
 ‘tvec__stop_words’: [None, ‘english’],

}# Setting up randomforest params
rf_params = {
 ‘tvec__max_features’:[2000],
 ‘tvec__ngram_range’: [(1, 2)],
 ‘tvec__stop_words’: [‘english’],
 ‘rf__max_depth’: [1000],
 ‘rf__min_samples_split’: [100],
 ‘rf__max_leaf_nodes’: [None]
}
```

现在我们有了自己的模型，让我给你一个公平的**警告**。RandomForest 需要一些时间来运行，如果你有合适的计算能力，我建议只运行一个模型。

P 参数必须被指定给一个模型，并被构建成一个带有键和值的字典。这里的方法是 RandomForest 的模型变量是“rf ”,我们需要在变量和参数之间使用双下划线“__”。例如，如果我们想在 RandomForest 中设置叶节点的数量，我们需要在我们的参数中写出来，就像这样:“RF _ _ max _ leaf _ nodes”:[None]”。为了避免出错，我们需要使用方括号'[ ]'以列表形式传递每个字典值，即使只有一个。您不需要调用每个参数，只需要调用您试图在模型中使用的那些参数。

# 网格搜索

为了设置 GridSearch，我们将使用我们构建的参数传递我们的管道。我们将对我们的五折模型进行标准交叉验证。

```
# Setting up GridSearch for Randomforest
rf_gs = GridSearchCV(rf_pipe, param_grid=rf_params, cv = 5, verbose = 1, n_jobs = -1)# Setting up GridSearch for TFIDFVectorizer
tvc_gs = GridSearchCV(tvc_pipe, param_grid=tf_params, cv = 5, verbose =1, n_jobs = -1)# Fitting TVC GS
tvc_gs.fit(X_train, y_train)# Fitting Randomforest CV GS
rf_gs.fit(X_train, y_train)
```

当使用 GridSearch 时，我们将管道作为估计器。然后我们必须给它一个 param_grid，这是我们在构建管道时设置的参数。建议您使用三重或五重交叉验证。我更喜欢使用五重交叉验证，但你可以为“cv”传递三个。

我建议在运行任何模型时设置“verbose = 1”和“n_jobs = -1”。Verbose 将告诉我们运行时间以及模型运行需要多长时间。n_jobs 参数允许我们指定我们希望在 CPU 上使用多少个内核。这可以减少运行模型的时间。在这里使用-1 将告诉它使用 CPU 上的所有内核。这个过程需要的时间最长，因为模型现在正在数据集上进行自我训练。然后，我们将使用它给出一个准确度分数，它将告诉我们我们的模型执行得有多好。

![](img/76e77c97bfc07fd17d8b3a0197fef631.png)

[Source](http://Photo by JOSHUA COLEMAN on Unsplash)

# 得分

既然我们已经构建并拟合了我们的模型，我们希望对训练数据和测试数据进行评分。你还记得我们之前把它分成了两个数据集吗？我们在训练数据集上“训练”或“拟合”了我们的模型。我们还将使用训练数据集对其进行评分，以查看它是否可以使用特征(X 变量)来确定我们的目标(y 变量)。

让我们给模型打分:

```
# Scoring Training data on TFIDFVectorizer
tvc_gs.score(X_train, y_train)#score: 0.8742193813827052# Scoring Test data on TFIDFVectorizer
tvc_gs.score(X_test, y_test)#score: 0.8627148523578669# Scoring Training data on RandomForest
rf_gs.score(X_train, y_train)#score: 0.9380648005289839# Checking Test score on RandomForest
rf_gs.score(X_test, y_test)#score: 0.881004847950639
```

貌似两款都表现不错！事实上，TFIDF 矢量器的性能优于 RandomForest。当我们查看分数时，我们希望训练分数和测试分数尽可能接近。在这里，RandomForest 的训练和测试分数之间的距离更大，而我们的 TFIDF 矢量器的分数几乎相同。我们可以看看这两个模型，看到 RandomForest 由于分数的不同而过度拟合。我们希望选择 TFIDF 矢量器作为用于预测的模型。

# 重要

所以我们知道哪种型号性能最好。它用来做那些预测的词呢？当使用管道时，我们需要“进入”估算器，从那里我们可以看到在进行预测时它发现什么特征(词)是最重要的。让我们用这个做一个数据框列，按最重要的排序。

```
tvc_title = pd.DataFrame(rf_pipe.steps[1][1].feature_importances_, tvc_pipe.steps[0][1].get_feature_names(), columns=[‘importance’])tvc_title.sort_values(‘importance’, ascending = False).head(20)
```

上面的代码将为我们提供一个数据框架，其中包含我们的模型用来进行预测的前 20 个最重要的单词。从这里，我们可以创建一些可视化来更好地表示我们的模型。

# 结论

根据上面的信息，我们可以得出结论，利用自然语言处理的能力，我们可以预测一篇帖子可能位于哪个博客主题，准确率为 88%。这怎么可能有用呢？从商业的角度来看，我们希望知道一个帖子是否会出现在我们的目标主题中，这样我们就可以关注最有可能看到该帖子的受众。我们还可以显示在生成预测时最重要的特征，并查看每个特征对我们预测的影响程度。