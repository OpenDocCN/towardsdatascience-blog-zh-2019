# 使用深度学习和位置敏感哈希查找相似图像

> 原文：<https://towardsdatascience.com/finding-similar-images-using-deep-learning-and-locality-sensitive-hashing-9528afee02f5?source=collection_archive---------3----------------------->

## 使用 FastAI & Pytorch 通过 ResNet 34 的图像嵌入找到相似图像的简单演练。也在巨大的图像嵌入集合中进行快速语义相似性搜索。

![](img/18a456cda0be80b6a9ce669385fa4283.png)

Fina output with similar images given an Input image in Caltech 101

在这篇文章中，我们试图实现上述结果，即给定一幅图像，我们应该能够从 [Caltech-101 数据库](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)中找到相似的图像。这篇文章用一个端到端的过程来指导我如何构建它。复制项目的全部代码都在我的 [GitHub 库](https://github.com/aayushmnit/Deep_learning_explorations/tree/master/8_Image_similarity_search)中。实现上述结果的过程可以分为以下几个步骤-

1.  使用 [FastAI](http://docs.fast.ai/) 和 [Pytorch](https://pytorch.org/) 从 ResNet-34 模型(在 ImageNet 上训练)转移学习以检测 [Caltech-101 数据集](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)中的 101 个类。
2.  从已训练的 ResNet 34 模型中取倒数第二个完全连接的层的输出，以得到所有 9，144 个 Caltech-101 图像的嵌入。
3.  使用位置敏感散列来为我们的图像嵌入创建 LSH 散列，这使得快速近似最近邻搜索成为可能
4.  然后给定一个图像，我们可以使用我们训练的模型将其转换为图像嵌入，然后在 Caltech-101 数据集上使用近似最近邻搜索相似的图像。

# 第 1 部分—数据理解和迁移学习

正如我上面提到的，对于这个项目，我的目标是查询任何给定的图像，并在 [Caltech-101 数据库](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)中找到语义相似的图像。该数据库包含 9，144 幅图像，分为 101 个类别。每个类别大约有 50-800 张图片。

![](img/ca7976bf22187372548bd4ccec1c4fb6.png)

Image examples from Caltech-101 database

我们项目的第一个练习是获得一个深度学习网络，它可以准确地对这些类别进行分类。对于此任务，我们将使用预训练的 ResNet 34 网络，该网络在 ImageNet 数据库上进行训练，并使用 Pytorch 1.0 和 FastAI 库对 Caltech-101 数据库的 101 个类别进行分类。正如我在我的前一篇博客中所写的，我将在这篇博客中概述这个过程。你可以参考[这本笔记本](https://github.com/aayushmnit/Deep_learning_explorations/blob/master/8_Image_similarity_search/Image%20similarity%20on%20Caltech101%20using%20FastAI%2C%20Pytorch%20and%20Locality%20Sensitive%20Hashing.ipynb)找到代码做同样的事情。找到下面的步骤做迁移学习分类加州理工学院-101 图像-

1.  使用 FastAI 库使用 Pytorch 的数据集加载器加载数据
2.  采用预训练的网络，在这种情况下，ResNet 34，并删除其最后完全连接的层
3.  在网络末端添加新的完全连接的层，并仅使用 Caltech-101 图像训练这些层，同时保持所有其他层冻结
4.  通过解冻所有层来训练整个网络

# 第 2 部分—使用 Pytorch 钩子提取图像嵌入

现在我们有了一个预先训练好的网络，我们需要从这个网络中为我们所有的 Caltech-101 图像提取嵌入。嵌入只不过是一个对象在 N 维向量中的表示。在这种情况下，图像嵌入是图像在 N 维中的表示。基本思想是给定图像与另一图像越接近，它们的嵌入在空间维度上也将是相似和接近的。

![](img/e360682cd466dc532d8c4187178cc1cc.png)

Image embedding visualization. Credit — [Blog](https://medium.com/gsi-technology/visualising-embeddings-using-t-sne-8fd4e31b56e2)

你可以在上面这张取自[博客](https://medium.com/gsi-technology/visualising-embeddings-using-t-sne-8fd4e31b56e2)的图片中看到，图像嵌入是一种矢量化形式的图像空间表示，其中相似的图像在空间维度上也很接近。

我们可以从 ResNet-34 获得图像嵌入，方法是取其倒数第二个全连接层的输出，该层的维数为 512。为了在 Pytorch 的深度学习模型中保存中间计算以供检查，或者在我们的情况下提取嵌入，我们使用 Pytorch 钩子。钩子有两种类型——向前和向后。前向钩子用于保存在网络中向前传递的信息以进行推断，而后向钩子用于在反向传播期间收集关于梯度的信息。在我们的例子中，我们需要在推理阶段输出倒数第二个完全连接的层，这意味着我们需要使用一个前向钩子。让我们来看看创建钩子的代码(也在我的[笔记本](https://github.com/aayushmnit/Deep_learning_explorations/blob/master/8_Image_similarity_search/Image%20similarity%20on%20Caltech101%20using%20FastAI%2C%20Pytorch%20and%20Locality%20Sensitive%20Hashing.ipynb)的“提取特征”部分)

```
**class** **SaveFeatures**():
    features=**None**
    **def** __init__(self, m): 
        self.hook = m.register_forward_hook(self.hook_fn)
        self.features = **None**
    **def** hook_fn(self, module, input, output): 
        out = output.detach().cpu().numpy()
        **if** isinstance(self.features, type(**None**)):
            self.features = out
        **else**:
            self.features = np.row_stack((self.features, out))
    **def** remove(self): 
        self.hook.remove()
```

创建 Pytorch 钩子只需要上面的代码。SaveFeatures 类从[的 torch.nn 模块](https://pytorch.org/docs/stable/nn.html)调用 [register_forward_hook](https://pytorch.org/docs/stable/nn.html?highlight=register_forward_hook#torch.nn.Module.register_forward_hook) 函数，并给定任何模型层，它将把中间计算保存在 numpy 数组中，该数组可以使用 SaveFeatures.features 函数检索。让我们看看使用这个类的代码—

```
*## Output before the last FC layer*
sf = SaveFeatures(learn.model[1][5])*## By running this feature vectors would be saved in sf variable initated above*
_= learn.get_preds(data.train_ds)
_= learn.get_preds(DatasetType.Valid)## Converting in a dictionary of {img_path:featurevector}
img_path = [str(x) **for** x **in** (list(data.train_ds.items)+list(data.valid_ds.items))]
feature_dict = dict(zip(img_path,sf.features))
```

第 1–2 行:调用类 SaveFeatures，使用模型层引用将倒数第二个全连接层的输出作为输入。

第 4–6 行:传递 Caltech-101 数据以获得他们的预测。请注意，我们对保存预测不感兴趣，这就是我们使用“_”的原因在这种情况下，倒数第二层的中间输出保存在名为“sf”的变量中，该变量是 SaveFeatures 类的一个实例。

第 8–10 行:创建一个 python 字典，其中图像路径是键，图像嵌入是值。

现在我们的字典中有了 Caltech-101 中每个图像的嵌入表示。

# 第 3 部分—快速近似最近邻搜索的位置敏感散列法

我们可以使用我们新生成的加州理工学院 101 图像嵌入，并获得一个新的图像，将其转换为嵌入，以计算新图像和所有加州理工学院 101 数据库的距离 b/w，以找到类似的图像。这个过程本质上是计算昂贵的，并且作为新的图像嵌入，必须与加州理工学院 101 数据库中的所有 9K+图像嵌入进行比较，以找到最相似的图像(最近邻)，这在计算复杂度符号中是 O(N)问题，并且随着图像数量的增加，将花费指数更多的时间来检索相似的图像。

为了解决这个问题，我们将使用位置敏感哈希(LSH ),这是一种近似的最近邻算法，可以将计算复杂度降低到 O(log N)。[这篇博客](/fast-near-duplicate-image-search-using-locality-sensitive-hashing-d4c16058efcb)从时间复杂度和实现方面很详细地解释了 LSH。简而言之，LSH 为图像嵌入生成一个哈希值，同时牢记数据的空间性；特别是；高维度相似的数据项将有更大的机会接收到相同的哈希值。

下面是 LSH 如何转换大小为 K-的散列中的嵌入的步骤

1.  在嵌入维中生成 K 个随机超平面
2.  检查特定嵌入是在超平面之上还是之下，并指定 1/0
3.  对每个 K 超平面执行步骤 2，以获得哈希值

![](img/bb0193137bdb682db85fe576ea1b6bef.png)

the hash value of the orange dot is 101 because it: 1) above the purple hyperplane; 2) below the blue hyperplane; 3) above the yellow hyperplane. Image Credit — [Link](/fast-near-duplicate-image-search-using-locality-sensitive-hashing-d4c16058efcb)

现在让我们看看 LSH 将如何执行人工神经网络查询。给定一个新的图像嵌入，我们将使用 LSH 为给定的图像创建一个散列，然后比较来自共享相同散列值的 Caltech-101 数据集的图片的图像嵌入的距离。以这种方式，代替在整个 Caltech-101 数据库上进行相似性搜索，我们将仅对与输入图像共享相同散列值的图像子集进行相似性搜索。对于我们的项目，我们使用 [lshash3](https://pypi.org/project/lshash3/) 包进行近似最近邻搜索。让我们看看做同样事情的代码(你可以在我的[笔记本](https://github.com/aayushmnit/Deep_learning_explorations/blob/master/8_Image_similarity_search/Image%20similarity%20on%20Caltech101%20using%20FastAI%2C%20Pytorch%20and%20Locality%20Sensitive%20Hashing.ipynb)的“使用位置敏感散列法查找相似图片”部分找到代码)

```
**from** **lshash** **import** LSHash

k = 10 *# hash size*
L = 5  *# number of tables*
d = 512 *# Dimension of Feature vector*
lsh = LSHash(hash_size=k, input_dim=d, num_hashtables=L)*# LSH on all the images*
**for** img_path, vec **in** tqdm_notebook(feature_dict.items()):
    lsh.index(vec.flatten(), extra_data=img_path)
```

上面的代码获取图像嵌入字典，并将其转换为 LSH 表。要查询 LSH 表，我们可以使用下面的代码—

```
# query a vector q_vec
response = lsh.query(q_vec, num_results= 5)
```

# 第 4 部分—将所有内容放在一起

现在我们已经创建了 LSH 表，让我们编写一个脚本，它可以将图像 URL 作为输入，并从加州理工 101 数据库中给我们 N 个(用户定义的)相似图像。这部分的代码在我的 [Github 这里](https://github.com/aayushmnit/Deep_learning_explorations/blob/master/8_Image_similarity_search/find_similar_image.py)。

![](img/c2d0d7450060bb1d2d26ae898bf055d5.png)

Process flow of the [find similar image](https://github.com/aayushmnit/Deep_learning_explorations/blob/master/8_Image_similarity_search/find_similar_image.py) script.

该脚本执行以下任务-

1.  加载 LSH 表和我们的 ResNet 34 模型(load_model 函数)
2.  从用户调用中获取图像 URL 并下载图像(download_img_from_url 函数)
3.  从 ResNet-34 传递图像以获得 512 维图像嵌入(image_to_vec 函数)
4.  用 LSH 表查询，找到 N 个(自定义)相似图像及其路径(get_similar_images 函数)
5.  在所需的输出路径返回输出，也可以使用 Open CV (get_similar_images 函数)显示输出

我们可以在各种应用程序中使用类似的概念，如在我们的照片库中查找类似的图像，相似外观项目的项目-项目推荐，对图像进行网络搜索，查找近似重复的图像等。

**总结(TL；博士)**。

在博客中，我们看到了深度学习在寻找*语义相似图像*中的应用，以及如何使用位置敏感哈希(LSH)进行*近似最近邻*查询，以加快大型数据集的查询时间。此外，值得注意的是，我们不是在原始特征(图像)上使用 LSH，而是在嵌入上，这有助于在庞大的集合中进行*快速相似性搜索。*

我希望你喜欢阅读，并随时使用我在 [Github](https://github.com/aayushmnit/Deep_learning_explorations/tree/master/7_Facial_attributes_fastai_opencv) 上的代码来为你的目的进行测试。此外，如果对代码或博客文章有任何反馈，请随时联系 LinkedIn 或发电子邮件给我，地址是 aayushmnit@gmail.com。您也可以在 Medium 和 Github 上关注我，了解我将来可能会写的博客文章和探索项目代码。