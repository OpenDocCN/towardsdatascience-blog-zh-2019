# 上传大文件到 GitHub

> 原文：<https://towardsdatascience.com/uploading-large-files-to-github-dbef518fa1a?source=collection_archive---------1----------------------->

## 3 种避免在上传过程中收到错误信息的方法

![](img/c07af49159164f22731ee7d719c5df69.png)

Photo by [Jay Wennington](https://unsplash.com/@jaywennington?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

GitHub 有严格的 100MB 文件限制。如果你只是上传几行代码，这不是你需要担心的事情。然而，如果你想上传一点数据，或者二进制的东西，这是一个你可能想要跨越的限制。这里有三种不同的方法来克服 **100MB 的限制**。

*原载于我的博客*[*edenau . github . io*](https://edenau.github.io)*。*

# 1.。gitignore

创建一个文件 ***。gitignore*** 在存储库的父目录中，存储所有希望 Git 忽略的文件目录。使用`*`作为通配符，这样您就不需要在每次创建新的大文件时手动添加文件目录。这里有一个例子:

```
*.nc
*.DS_store
```

这些被忽略的文件会被 Git 自动忽略，不会上传到 GitHub。不再有错误消息。

# 2.仓库清理器

如果你不小心在本地提交了超过 100MB 的文件，你将很难把它推送到 GitHub。它不能通过删除大文件并再次提交来解决。这是因为 GitHub 跟踪每一次提交，而不仅仅是最近一次。从技术上讲，您是在整个提交记录中推送文件。

![](img/418e6f452aaee3fadb5d851d23ed1d8f.png)

Photo by [Samuel Zeller](https://unsplash.com/@samuelzeller?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

虽然您可以通过分支在技术上解决它，但这绝不是简单的。幸运的是，您可以运行一个 ***库清理器*** ，它会自动清理所有的大文件提交。

下载 [BFG 回购清理器](https://rtyley.github.io/bfg-repo-cleaner/) *bfg.jar* 并运行以下命令:

```
java -jar [bfg.jar](https://rtyley.github.io/bfg-repo-cleaner/#download) --strip-blobs-bigger-than 100M <your_repo>
```

它会自动清理您的提交，并生成一个新的提交，注释为“删除大文件”。按下它，你就可以走了。

# 3.吉特 LFS

你可能已经注意到，上述两种方法都避免上传大文件。如果您真的想上传它们，以便在另一台设备上访问它们，该怎么办？

Git 大文件存储让你把它们存储在一个远程服务器上，比如 GitHub。将 git-lfs 放入您的 *$PATH* 中下载并安装 ***。然后，您需要对每个本地存储库***运行一次下面的命令**:**

```
git lfs install
```

大文件由以下人员选择:

```
git lfs track '*.nc'
git lfs track '*.csv'
```

这会创建一个名为 ***的文件。gitattributes*** ，瞧！您可以正常执行添加和提交操作。然后，你首先需要 a)将文件推送到 LFS，然后 b)将**指针**推送到 GitHub。以下是命令:

```
git lfs push --all origin master
git push -u origin master
```

![](img/9083df47baf5374b8471f1234c54d0c7.png)

Photo by [Lucas Gallone](https://unsplash.com/@lucasgallone?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

Git LFS 上的文件可以在 GitHub 上找到，标签如下。

![](img/87d9f2233081796999f8783f8efacec6.png)

为了将存储库拉到另一个设备上，只需在那个设备上安装 ***git-lfs*** (每个本地存储库)。

## 相关文章

感谢您的阅读！如果您对数据科学感兴趣，请查看以下文章:

[](/5-python-features-i-wish-i-had-known-earlier-bc16e4a13bf4) [## 我希望我能早点知道的 5 个 Python 特性

### 超越 lambda、map 和 filter 的 Python 技巧

towardsdatascience.com](/5-python-features-i-wish-i-had-known-earlier-bc16e4a13bf4) [](/why-sample-variance-is-divided-by-n-1-89821b83ef6d) [## 为什么样本方差除以 n-1

### 解释你的老师没有教过的高中统计学

towardsdatascience.com](/why-sample-variance-is-divided-by-n-1-89821b83ef6d) 

*原载于我的博客*[*edenau . github . io*](https://edenau.github.io)*。*