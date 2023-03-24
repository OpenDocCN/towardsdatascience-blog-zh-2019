# 数据科学家的 GIT 存储和分支

> 原文：<https://towardsdatascience.com/git-stashing-and-branching-for-data-scientists-21ef6e0fb68?source=collection_archive---------34----------------------->

## 数据科学项目中的连续过程

![](img/71dfc5b32fe128abfa5cfb3fad94cae5.png)

Photo by [Kristopher Roller](https://unsplash.com/@krisroller?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/branches?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

> 数据科学家，分析数据，做预处理，建立模型，测试模型，部署模型。但是在所有这些步骤中，肯定应该用 R 或 Python 编程。通常一个人或一组人在同一个项目上工作，这需要整合，这样每个人都应该在同一页上。为了进行这种集成，版本控制成为大多数数据科学家使用 GIT 作为版本控制的一种方式。

在这篇文章中，我将关注我们在 GIT 上面临的主要挑战。它们主要是存储和分支。

# 隐藏

让我们假设您正在构建一个模型，并且收到了另一个变更请求。您需要快速保存当前的变更，并在其他变更请求工作完成后将它们应用到存储库中。

> 为了做到这一点，GIT 提供了一个有用的命令 **git stash** ，它允许你保存你的更改并返回到干净的工作目录。

## 创建一个仓库

假设您已经完成了数据预处理，并且想要临时保存您的更改并返回到工作目录。下面的命令将帮助你做到这一点。

```
git stash “preprocessing done”
```

## 列出所有藏匿处

您刚刚创建了一个仓库，如果您想查看所有现有的仓库，可以使用下面的命令。

```
git stash list
```

> 输出

```
stash@{0}: preprocessing done
stash@{1}: get the data
```

最新的存储将在顶部，索引为 0。

## 应用存储

到目前为止，您已经创建了 stash，并假设您已经完成了新的工作或者收到了变更请求。现在您必须取回您保存的更改，这可以通过应用创建的存储来完成。

```
git stash apply stash@{0}
```

这将应用更改，但存储不会从列表中删除。为此，我们必须使用下面的命令。

```
git stash drop stash@{0}
```

# 分支

数据科学家永远不会以构建单一模型而告终，而是继续尝试其他预处理技术或算法。因此，为了不干扰现有的分支，我们去创建分支。当工作是实验性的或者添加新的特性来封装变更时，我们创建分支。

在这里，我将指导您如何从现有的分支创建一个分支，删除一个分支，列出所有分支以及合并分支。

## 列出所有分支

在我们开始创建和删除分支之前，让我们尝试用下面的命令列出所有现有的分支。

```
git branch -a
```

> **-a** 指定我们希望看到本地系统和远程存储库中的所有分支。

## 从主控形状创建分支

要从主服务器创建本地分支，可以使用以下命令。

```
git checkout -b new-branch master
```

您可以添加、修改、删除任何文件并提交更改。

```
git add <filename> git commit -m “New features created”
```

如果您想将这个分支推到远程，可以使用下面的命令。

```
git push origin new-branch
```

## 合并分支

到目前为止，我们已经创建了新的分支，假设我们已经在新的分支中实现了一些特性。下一个任务是将这个分支合并到主分支中。我们可以使用下面的命令来实现这一点。

```
git checkout master 
git merge new-branch
```

> **结帐主**，指定您正在切换到主分支

稍后您可以将这些更改推送到存储库中。

## 删除本地和远程分支

数据科学项目包含许多功能和变更请求。我们不断地创造我们想要的分支，它们会堆积起来。为了摆脱它们，我们需要删除它们。

**删除一个本地分支**

```
git branch -d <branch name>
```

使用上面的命令，分支将在您的本地系统中被删除，但不会在远程系统中被删除。

**删除一个远程分支**

```
git push <remote name> :<branch name>
```

上述命令将删除存储库中的分支。

N **注意:**我从主节点获取了创建或合并分支，但是它可以是任何现有的分支。

> 希望你喜欢它！！敬请期待！！！请对任何疑问或建议发表评论！！！