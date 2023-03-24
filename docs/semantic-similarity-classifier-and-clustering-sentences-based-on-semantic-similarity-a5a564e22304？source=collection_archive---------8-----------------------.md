# 语义相似度分类器和基于语义相似度的句子聚类。

> 原文：<https://towardsdatascience.com/semantic-similarity-classifier-and-clustering-sentences-based-on-semantic-similarity-a5a564e22304?source=collection_archive---------8----------------------->

![](img/24a9259f078d96b3efe6bc0389719e81.png)

最近，我们一直在做一些实验，通过利用预先训练的模型来聚集语义相似的消息，这样我们就可以在不使用标记数据的情况下获得一些东西。这里的任务是给定一个句子列表，我们对它们进行聚类，使得语义相似的句子在同一个聚类中，并且聚类的数量不是预先确定的。

语义相似度分类器的任务是:对给定的两个句子/消息/段落进行语义等价的分类。

## 第一步:通过嵌入来表示每个句子/信息/段落。对于这个任务，我们使用了 infersent，它工作得很好。

```
*InferSent* is a *sentence embeddings* method that provides semantic representations for English sentences. It is trained on natural language inference data and generalizes well to many different tasks.
```

[](https://github.com/facebookresearch/InferSent) [## Facebook 研究/推断

### 推断句嵌入。在 GitHub 上创建一个帐户，为 Facebook research/INF sent 开发做贡献。

github.com](https://github.com/facebookresearch/InferSent) 

代码如下:

```
*# Load infersent model* 
model_version = 2
MODEL_PATH = "infersent_sentence_encoder/infersent**%s**.pkl" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))*# If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.*
W2V_PATH = 'infersent_sentence_encoder/GloVe/glove.840B.300d.txt' **if** model_version == 1 **else** 'infersent_sentence_encoder/fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)#load data
ds = pd.read_msgpack('./ds.mp')
sentences = ds['text']# generate infersent sentence embeddings
model.build_vocab(sentences, tokenize=**True**)
embs = model.encode(sentences, tokenize=**True**)
```

## 步骤 2:寻找语义相似的句子/信息/段落的候选

这里的想法是索引每个句子/消息/段落的表示(嵌入),并基于距离阈值为每个句子挑选 k (=10)个 NN(最近邻)候选。我们发现 nmslib 非常快速高效。

```
**import** **nmslib**

NTHREADS = 8
**def** create_index(a):
    index = nmslib.init(space='angulardist')
    index.addDataPointBatch(a)
    index.createIndex()
    **return** index**def** get_knns(index, vecs, k=3):
    **return** zip(*index.knnQueryBatch(vecs, k=k,num_threads=NTHREADS))nn_wvs = create_index(embs)to_frame = **lambda** x: pd.DataFrame(np.array(x)[:,1:])idxs, dists = map(to_frame, get_knns(nn_wvs, embs, k=10))catted = pd.concat([idxs.stack().to_frame('idx'), dists.stack().to_frame('dist')], axis=1).reset_index().drop('level_1',1).rename(columns={'level_0': 'v1', 'idx': 'v2'})
```

## 第三步:获得候选对在语义相似度分类器上的预测概率。(关于语义相似性分类器的细节将在以后的博客文章中介绍)

把第二步想成候选生成(侧重于召回)，第三步想成侧重于精度。在所有被认为是潜在重复的候选者中，我们给每一对分配概率。

## 步骤 4:聚集聚类以合并聚类

基于在步骤 3 中被认为是重复的候选者，我们使用 scikit 中的凝聚聚类实现来合并聚类。在凝聚聚类中，所有观察值都从它们自己的聚类开始，并且使用指定的合并标准合并聚类，直到收敛，此时不再发生合并。

[](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) [## sk learn . cluster . agglomerate clustering-sci kit-learn 0 . 21 . 2 文档

### 连通性矩阵。为每个样本定义遵循给定数据结构的相邻样本。这可以…

scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) 

这在实践中工作得相当好，并且形成的集群具有良好的语义等价性。