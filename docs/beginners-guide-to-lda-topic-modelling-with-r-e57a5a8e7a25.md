# 用 R 进行 LDA 主题建模的初学者指南

> 原文：<https://towardsdatascience.com/beginners-guide-to-lda-topic-modelling-with-r-e57a5a8e7a25?source=collection_archive---------0----------------------->

## 识别非结构化文本中的主题

![](img/3f809b91d1ee5c983f9e00f7b56dafde.png)

[Picture credits](http://www.contrib.andrew.cmu.edu)

现在许多人想从自然语言处理(NLP)开始。然而，他们不知道从哪里以及如何开始。这可能是因为有太多的“指南”或“读物”可用，但它们并没有确切地告诉你从哪里以及如何开始。这篇文章旨在给读者一步一步的指导，告诉他们如何使用 r 的**潜在狄利克雷分配(LDA)** 分析进行主题建模。

这种技术很简单，在小数据集上很有效。因此，我建议第一次尝试 NLP 和使用主题建模的人使用这种技术。

什么是主题建模？通俗地说，主题建模是试图在不同的文档中找到相似的主题，并试图将不同的单词组合在一起，这样每个主题将由具有相似含义的单词组成。我经常喜欢打一个比方——当你有一本故事书被撕成不同的书页时。在你尝试运行一个主题建模算法之后，你应该能够提出各种各样的主题，这样每个主题将由来自每个章节的单词组成。否则，你可以简单地使用情绪分析——正面或负面评论。

> 在[机器学习](https://en.wikipedia.org/wiki/Machine_learning)和[自然语言处理](https://en.wikipedia.org/wiki/Natural_language_processing)中，**主题模型**是一种[统计模型](https://en.wikipedia.org/wiki/Statistical_model)，用于发现文档集合中出现的抽象“主题”。——[维基百科](https://en.wikipedia.org/wiki/Topic_model)

在正式介绍了主题建模之后，文章的剩余部分将描述如何进行主题建模的逐步过程。它由 4 个部分组成:数据加载、数据预处理、建立模型和主题中单词的可视化。

如上所述，我将使用 LDA 模型，这是一个概率模型，它为 word 分配一个最有可能属于它的主题的概率分数。我将跳过 LDA 的技术解释，因为有许多可用的文章。(例:这里[这里](https://www.tidytextmining.com/topicmodeling.html))不用担心，我会解释所有的术语，如果我使用它的话。

1.  **数据加载**

为了简单起见，我们将使用的数据集将是来自 [kaggle](#https://www.kaggle.com/crowdflower/first-gop-debate-twitter-sentiment) 的前 5000 行 twitter 情绪数据。对于我们的模型，我们不需要标记数据。我们所需要的只是一个我们想要从中创建主题的文本列和一组唯一的 id。最初有 18 列和 13000 行数据，但是我们将只使用 text 和 id 列。

```
data <- fread(“~/Sentiment.csv”)
#looking at top 5000 rows
data <- data %>% select(text,id) %>% head(5000)
```

![](img/8a10f84c250c6dfc3ab5c435eabbaf3e.png)

Dataframe after selecting the relevant columns for analysis

**2。预处理**

正如我们从文本中观察到的，有许多 tweets 由不相关的信息组成:如 RT、twitter 句柄、标点符号、停用词(and、or the 等)和数字。这些会给我们的数据集增加不必要的噪声，我们需要在预处理阶段去除这些噪声。

```
data$text <- sub("RT.*:", "", data$text)
data$text <- sub("@.* ", "", data$text)text_cleaning_tokens <- data %>% 
  tidytext::unnest_tokens(word, text)
text_cleaning_tokens$word <- gsub('[[:digit:]]+', '', text_cleaning_tokens$word)
text_cleaning_tokens$word <- gsub('[[:punct:]]+', '', text_cleaning_tokens$word)
text_cleaning_tokens <- text_cleaning_tokens %>% filter(!(nchar(word) == 1))%>% 
  anti_join(stop_words)
tokens <- text_cleaning_tokens %>% filter(!(word==""))
tokens <- tokens %>% mutate(ind = row_number())
tokens <- tokens %>% group_by(id) %>% mutate(ind = row_number()) %>%
  tidyr::spread(key = ind, value = word)
tokens [is.na(tokens)] <- ""
tokens <- tidyr::unite(tokens, text,-id,sep =" " )
tokens$text <- trimws(tokens$text)
```

**3。模型构建**

有趣的部分终于来了！创建模型。

首先，您必须创建一个 DTM(文档术语矩阵)，这是一个稀疏矩阵，包含您的术语和文档作为维度。构建 DTM 时，您可以选择如何对文本进行分词(将一个句子拆分成一个或两个单词)。这将取决于你希望 LDA 如何解读你的话。你需要问问自己，在你的上下文中，单数词或双数词(短语)是否有意义。例如，如果你的文本包含许多单词，如“执行失败”或“不赞赏”，那么你将不得不让算法选择一个最多 2 个单词的窗口。否则，使用单字也可以。在我们的例子中，因为它是 Twitter 情感，我们将使用 1-2 个单词的窗口大小，并让算法为我们决定哪些是更重要的短语连接在一起。我们还将探索术语频率矩阵，它显示了单词/短语在整个文本语料库中出现的次数。如果该项小于 2 次，我们丢弃它们，因为它不会给算法增加任何值，并且它也将有助于减少计算时间。

```
#create DTM
dtm <- CreateDtm(tokens$text, 
                 doc_names = tokens$ID, 
                 ngram_window = c(1, 2))#explore the basic frequency
tf <- TermDocFreq(dtm = dtm)
original_tf <- tf %>% select(term, term_freq,doc_freq)
rownames(original_tf) <- 1:nrow(original_tf)# Eliminate words appearing less than 2 times or in more than half of the
# documents
vocabulary <- tf$term[ tf$term_freq > 1 & tf$doc_freq < nrow(dtm) / 2 ]dtm = dtm
```

使用 DTM，您可以运行 LDA 算法进行主题建模。你将不得不手动分配多个主题 k。接下来，算法将计算一个连贯性分数，以允许我们从 1 到 k 中选择最佳主题。什么是连贯性和连贯性分数？连贯性给出了每个主题的概率连贯性。连贯得分是计算同一主题中的单词放在一起时是否有意义的得分。这给了我们正在制作的主题的质量。特定数字 k 的分数越高，就意味着每个主题的相关词越多，主题就越有意义。例如:{狗，说话，电视，书}对{狗，球，吠，骨头}。后者将产生比前者更高的一致性分数，因为单词更紧密相关。

在我们的例子中，我们设置 k = 20 并对其运行 LDA，并绘制一致性分数。由分析师决定他们想要多少主题。

```
k_list <- seq(1, 20, by = 1)
model_dir <- paste0("models_", digest::digest(vocabulary, algo = "sha1"))
if (!dir.exists(model_dir)) dir.create(model_dir)model_list <- TmParallelApply(X = k_list, FUN = function(k){
  filename = file.path(model_dir, paste0(k, "_topics.rda"))

  if (!file.exists(filename)) {
    m <- FitLdaModel(dtm = dtm, k = k, iterations = 500)
    m$k <- k
    m$coherence <- CalcProbCoherence(phi = m$phi, dtm = dtm, M = 5)
    save(m, file = filename)
  } else {
    load(filename)
  }

  m
}, export=c("dtm", "model_dir")) # export only needed for Windows machines#model tuning
#choosing the best model
coherence_mat <- data.frame(k = sapply(model_list, function(x) nrow(x$phi)), 
                            coherence = sapply(model_list, function(x) mean(x$coherence)), 
                            stringsAsFactors = FALSE)
ggplot(coherence_mat, aes(x = k, y = coherence)) +
  geom_point() +
  geom_line(group = 1)+
  ggtitle("Best Topic by Coherence Score") + theme_minimal() +
  scale_x_continuous(breaks = seq(1,20,1)) + ylab("Coherence")
```

![](img/4162f6ea7b14093a83213c5fd93e4b8a.png)

在绘制 k 时，我们意识到 k = 12 给出了最高的一致性分数。在这种情况下，即使一致性分数相当低，也肯定需要调整模型，例如增加 k 以实现更好的结果或具有更多文本。但是出于解释的目的，我们将忽略该值，只使用最高的一致性分数。在了解了主题的最佳数量后，我们想看一下主题中的不同单词。每个主题将为每个单词/短语分配一个 phi 值(pr(word|topic)) —给定主题的单词的概率。所以我们只考虑每个主题中每个单词的前 20 个值。前 20 个术语将描述主题的内容。

```
model <- model_list[which.max(coherence_mat$coherence)][[ 1 ]]
model$top_terms <- GetTopTerms(phi = model$phi, M = 20)
top20_wide <- as.data.frame(model$top_terms)
```

![](img/1f59167cbcb1cb96f0c214d24a6c56ef.png)

Preview of top 10 words for the first 5 topic. The first word implies a higher phi value

上图是 12 个题目中的前 5 个题目。单词按 phi 值的升序排列。排名越高，这个词就越有可能属于这个主题。似乎有一些重叠的话题。这取决于分析师来思考我们是否应该通过目测将不同的主题组合在一起，或者我们可以运行一个树形图来查看哪些主题应该被分组在一起。树形图使用 Hellinger 距离(两个概率向量之间的距离)来决定主题是否密切相关。例如，下面的树形图表明主题 10 和 11 之间有更大的相似性。

```
model$topic_linguistic_dist <- CalcHellingerDist(model$phi)
model$hclust <- hclust(as.dist(model$topic_linguistic_dist), "ward.D")
model$hclust$labels <- paste(model$hclust$labels, model$labels[ , 1])
plot(model$hclust)
```

![](img/a14718dc287f68c462091463ee1b874a.png)

**4。可视化**

我们可以创建单词云，根据概率来查看属于某个主题的单词。下面代表主题 2。由于“gopdebate”是 topic2 中最有可能出现的单词，因此其大小将是单词云中最大的。

```
#visualising topics of words based on the max value of phi
set.seed(1234)final_summary_words <- data.frame(top_terms = t(model$top_terms))
final_summary_words$topic <- rownames(final_summary_words)
rownames(final_summary_words) <- 1:nrow(final_summary_words)
final_summary_words <- final_summary_words %>% melt(id.vars = c("topic"))
final_summary_words <- final_summary_words %>% rename(word = value) %>% select(-variable)
final_summary_words <- left_join(final_summary_words,allterms)
final_summary_words <- final_summary_words %>% group_by(topic,word) %>%
  arrange(desc(value))
final_summary_words <- final_summary_words %>% group_by(topic, word) %>% filter(row_number() == 1) %>% 
  ungroup() %>% tidyr::separate(topic, into =c("t","topic")) %>% select(-t)
word_topic_freq <- left_join(final_summary_words, original_tf, by = c("word" = "term"))pdf("cluster.pdf")
for(i in 1:length(unique(final_summary_words$topic)))
{  wordcloud(words = subset(final_summary_words ,topic == i)$word, freq = subset(final_summary_words ,topic == i)$value, min.freq = 1,
             max.words=200, random.order=FALSE, rot.per=0.35, 
             colors=brewer.pal(8, "Dark2"))}dev.off()
```

![](img/1a3416dea4c0803189a62567e708a62b.png)

Word cloud for topic 2

**5。结论**

我们用 LDA 完成了这个简单的主题建模，并用 word cloud 实现了可视化。你可以参考我的 [github](https://github.com/tqx94/Text-Analytics_LDA) 了解整个脚本和更多细节。这不是一个完整的 LDA 教程，因为还有其他很酷的度量标准，但是我希望这篇文章能为你提供一个很好的指导，告诉你如何使用 LDA 开始 R 中的主题建模。我也强烈建议每个人也阅读其他种类的算法。我相信你不会觉得无聊的！

如果你认为我错过了什么，请随时给我留言。

快乐话题造型！

**参考文献:**

1.  [维基百科](https://en.wikipedia.org/wiki/Topic_model)
2.  泰勒娃娃， [LDA 主题造型](/lda-topic-modeling-an-explanation-e184c90aadcd) (2018)
3.  托马斯·琼斯，[主题建模](https://cran.r-project.org/web/packages/textmineR/vignettes/c_topic_modeling.html) (2019)