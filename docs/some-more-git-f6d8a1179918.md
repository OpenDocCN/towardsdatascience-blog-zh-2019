# 再来点饭桶

> 原文：<https://towardsdatascience.com/some-more-git-f6d8a1179918?source=collection_archive---------35----------------------->

## 看看在这篇关于 Git 中分支和合并的文章之前的 [Git 基础知识](https://medium.com/@shivangisareen/git-basics-ec81696be4e6)文章！

![](img/6fbab526858410664ca8bc6340830129.png)

Photo by [Markus Spiske](https://unsplash.com/@markusspiske?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/code?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

Git 的设计是为了促进同一个项目的并行/多个版本。这个工具叫做分支。分支有效地允许我们将项目带入多个不同的方向，而不是单一的线性进展。它让我们维护一个分支，即“主”分支，它拥有原始版本的代码，然后还有其他分支，允许我们处理原始代码的不同版本，然后，将这些分支合并到原始代码中——合并。

当有其他人在同一个项目上工作，但是在不同的特性上工作，同时发生但是分开开发时，这个工具特别有用。最后，我们可以将不同的分支(比如代表不同特性的不同分支)合并到主分支中。

## 与分支相关的命令:

*   列出当前在我们仓库中的所有分支。带有*和绿色的文本表示这是我们当前所在的分支。
*   `git branch -a`:列出所有远程和本地分支
*   `git branch <branch name>`:用该分支名称创建一个新分支
*   `git branch -d <branch name>`:删除分支
*   `git checkout <branch name>`:转移到不同的分支机构
*   `git commit -am "message"`:合并了添加和提交步骤——添加我们已经更改的所有行并提交它们。

要将两个分支合并在一起:转到(`git checkout`)分支`master`，然后:`git merge <name of branch to be merged with the master branch>`

注意:所有这些变化都发生在我们的本地计算机上。因此，主机站点只有主分支，而我们在我们的存储库中有一个新的分支——我们刚刚创建的分支。

如果我们**将**检出到新的分支，**添加**并**提交**对新分支所做的更改，然后**将**这些更改推送到主机站点，我们会得到一条消息，说明*“当前分支特性没有上游分支”*。这是因为我们在主机站点上没有可以推送的新分支—它只有一个主分支。

因此，为了推送至主机站点上的新分支，我们使用:

*   `git push --set-upstream origin <name of the new branch>`，其中 origin 是远程的名称(存储在主机站点上的存储库的版本)。因此，主机站点主分支本质上是，`origin/master`，而我们的本地主分支就是`master`分支。
*   `git fetch`:从源获取提交并下载到本地。因此，我们在本地得到的两个分支是，`master`和`origin/master`。

现在，如果我们希望我们的主分支反映整体的新变化，基本上，合并我们所在的位置，即`master`分支与`origin/master`所在的位置，我们:`git merge origin/master`。因此，我们的主分支现在反映了 origin 的最新版本。

## 叉

存储库的分支是该存储库的完全独立的版本，它只是原始版本的副本。分叉一个项目就是重命名它，启动一个新项目，围绕这个想法建立一个社区。

## **叉还是克隆？**

当我们分叉一个项目时，我们实际上只是在克隆它。fork 就是选择获取原项目的副本，启动父项目的不同线程。另一方面，贡献者可以选择克隆一个存储库，从而获得一个原始项目的副本来进行工作、修改、添加/删除。在做出他们认为合适的更改后，他们希望这些更改与代码的原始版本合并。为此，他们提交一个拉取请求。这仅仅是一种方式，说明“嘿，我已经做了一些更改，希望有人在将它们合并到代码的原始版本之前对它们进行审核”。这是一个非常有用的功能，可以获得持续的反馈和渐进的修改。

> *捐款时，将项目转入您自己的主机站点帐户是有益的。从那里:将其克隆到您的本地计算机，为您的新代码创建一个本地分支，进行更改，提交，推送，与 master 合并，删除本地分支，最后发送您的 pull 请求以查看您所做的更改是否生效！*

感谢您抽出时间阅读这篇文章！