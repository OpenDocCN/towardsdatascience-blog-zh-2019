# 为什么使用 Git 以及作为数据科学家如何使用 Git

> 原文：<https://towardsdatascience.com/why-git-and-how-to-use-git-as-a-data-scientist-4fa2d3bdc197?source=collection_archive---------3----------------------->

![](img/1ca29ad15666c2a6d008f70f2a9b97fd.png)

也许你在别的地方听说过 Git。

也许有人告诉你，Git 只面向软件开发人员，作为一名数据科学家，simple 对此毫不关心。

如果你是一名软件工程师转数据科学家，这个话题对你来说是非常熟悉的。

如果你是来自不同背景的有抱负的数据科学家，希望进入这个领域，这个主题是你会感兴趣的——无论是现在还是在不久的将来。

如果你已经是一名数据科学家，那么你就会知道我在这里写的是什么以及为什么。

在文章的最后，我希望我在 Git 方面的经验分享能够让**了解 Git** 的重要性，以及**作为一名数据科学初学者如何在您的数据科学工作中使用它。**

我们开始吧！

# 那么 Git 是什么？

> **Git** 是一个[分布式版本控制](https://en.wikipedia.org/wiki/Distributed_version_control)系统，用于在[软件开发](https://en.wikipedia.org/wiki/Software_development)期间跟踪[源代码](https://en.wikipedia.org/wiki/Source_code)的变更
> 
> —维基百科

看维基百科给出的这个定义，我也曾经处在你的位置，才觉得 Git 是为软件开发者做的。作为一名数据科学家，我与此无关，只是安慰自己。

事实上， [Git](https://www.atlassian.com/git/tutorials/what-is-git) 是当今世界上使用最广泛的现代版本控制系统。这是以分布式和协作的方式为项目(开源或商业)做贡献的最受认可和流行的方法。

除了分布式版本控制系统， [Git 的设计考虑了性能、安全性和灵活性](https://www.atlassian.com/git/tutorials/what-is-git)。

现在您已经理解了 Git 是什么，您脑海中的下一个问题可能是，“如果只有我一个人在做我的数据科学项目，它与我的工作有什么关系？”

而不能领会 Git 的重要性也是可以理解的(就像我上次做的那样)。直到我开始在真实世界环境中工作，我才如此感激学习和实践 Git，即使是在我独自从事个人项目的时候——在后面的部分你会知道为什么。

现在，请继续阅读。

# 为什么是 Git？

让我们来谈谈为什么。

为什么是 Git？

一年前，我决定学习 Git。我在 [GitHub](https://github.com/admond1994) 上第一次分享并发布了我的[模拟代码，这是我在 CERN](https://github.com/admond1994/Poisson-Superfish-Analysis) 为我的最后一年论文项目所做的。

同时很难理解 git 中常用的术语(Git 添加、提交、推送、拉取等。)，我知道这在数据科学领域很重要，成为开源代码贡献者的一员让我的数据科学工作比以往更有成就感。

于是我继续学习，不断“犯”。

当我加入目前的公司时，我在 Git 方面的经验派上了用场，在那里，Git 是不同团队之间代码开发和协作的主要方式。

更重要的是，当您的组织遵循敏捷软件开发框架时，Git 特别有用，在这种框架中，Git 的分布式版本控制使整个开发工作流更加高效、快速，并且易于适应变化。

我已经多次谈到版本控制。**那么版本控制到底是什么？**

版本控制是一个记录文件或文件集随时间变化的系统，这样你可以在以后调用特定的版本。

比方说，你是一名数据科学家，与一个团队合作，你和另一名数据科学家从事相同的工作，建立一个机器学习模型。酷毙了。

如果您对函数进行了一些更改，并上传到远程存储库，并且这些更改与[主分支](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell)合并，那么您的模型现在就变成了版本 1.1(只是一个例子)。另一位数据科学家也在 1.1 版中对相同的功能进行了一些更改，新的更改现在已合并到主分支中。现在模型变成了 1.2 版本。在任何时候，如果你的团队发现 1.2 版本在发布过程中有一些错误，他们可以随时调用之前的 1.1 版本。

> 这就是版本控制的美妙之处
> 
> —我

# 作为数据科学家如何使用 Git

![](img/0d8d73d0e77587228624e1230938d463.png)

[(Source)](https://unsplash.com/photos/IgUR1iX0mqM)

我们已经讨论了 Git 是什么以及它的重要性。

现在的问题归结为:**作为数据科学家如何使用 Git？**

要成为数据科学家，你不需要成为 Git 专家，我也不需要。这里的关键是**理解 Git 的工作流程**和**如何在日常工作中使用 Git。**

请记住，您不可能记住所有的 Git 命令。像其他人一样，在需要的时候随时可以谷歌一下。[足智多谋](/be-resourceful-one-of-the-most-important-skills-to-succeed-in-data-science-6ed5f33c2939)。

我将重点介绍在 Bitbucket 中使用 [Git(免费使用)。当然，这里的工作流程也适用于 GitHub。确切的说，我在这里使用的工作流是](https://www.atlassian.com/git/tutorials/learn-git-with-bitbucket-cloud) [Git 特性分支工作流](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow)，这是开源和商业项目常用的工作流。

如果您想了解更多这里使用的术语，[这是一个很好的起点](https://www.atlassian.com/git/glossary/terminology)。

## Git 功能分支工作流

特性分支工作流假设了一个中央存储库，`master`代表了正式的项目历史。

开发人员不是直接提交到他们的本地`master`分支，而是在每次开始工作一个新特性时创建一个新的分支。

特性分支可以(也应该)被推送到中央存储库。这使得在不接触任何官方代码的情况下与其他开发者共享一个特性成为可能——在这个例子中是`master`分支。

在您开始做任何事情之前，键入`git remote -v`以确保您的工作空间指向您想要使用的远程存储库。

## 1.从主分支开始，创建一个新分支

```
git checkout master
git pull
git checkout -b branch-name
```

假设`master`分支总是被维护和更新，您切换到本地`master`分支，并将最新的提交和代码拉到您的本地`master`分支。

让我们假设您想要创建一个本地分支，向代码中添加一个新特性，并在以后将更改上传到远程存储库。

一旦您获得了本地`master`分支的最新代码，让我们创建并签出一个名为`branch-name`的新分支，所有的更改都将在这个本地分支上进行。这意味着您当地的`master`分公司不会受到任何影响。

## 2.更新、添加、提交和推送您的更改到远程存储库

```
git status
git add <your-files>
git commit -m 'your message'
git push -u origin branch-name
```

好吧。这里发生了很多事情。让我们一个一个的分解。

一旦您做了一些更新，将新特性添加到您的本地`branch-name`中，并且您想要将更改上传到远程分支，以便稍后合并到远程`master`分支。

`git status`将因此输出所有的文件更改(跟踪或未跟踪)由你。在使用`git commit -m 'your message'`通过消息提交更改之前，您将使用`git add <your-files>`决定要暂存哪些文件。

在此阶段，您的更改仅出现在您的本地分支机构中。为了让您的更改出现在 Bitbucket 上的远程分支中，您需要使用`git push -u origin branch-name`来提交您的提交。

这个命令将`branch-name`推送到中央存储库(原点),`-u`标志将它添加为远程跟踪分支。在设置了跟踪分支之后，可以在没有任何参数的情况下调用`git push`来自动将新特性分支推送到 Bitbucket 上的中央存储库。

## 3.创建一个拉取请求，并对拉取请求进行更改

太好了！现在，您已经成功地添加了一个新特性，并将更改推送到您的远程分支。

您对自己的贡献感到非常自豪，并且希望在将远程分支与远程主分支合并之前获得团队成员的反馈。这使得其他团队成员有机会在变更成为主要代码库的一部分之前对其进行审查。

您可以[在 Bitbucket](https://confluence.atlassian.com/bitbucket/create-a-pull-request-to-merge-your-change-774243413.html) 上创建一个 pull 请求。

现在，您的团队成员已经查看了您的代码，并决定在将代码合并到主代码库— `master`分支之前，需要您做一些其他的更改。

```
git status
git add <your-files>
git commit -m 'your message'
git push
```

因此，您可以按照与之前相同的步骤进行更改、提交并最终将更新推送到中央存储库。一旦您使用了`git push`，您的更新将自动显示在 pull 请求中。就是这样！

如果其他人对您接触过的相同代码的目标进行了更改，您将会遇到合并冲突，这在正常的工作流中是很常见的。你可以在这里看到[关于如何解决合并冲突的](https://confluence.atlassian.com/bitbucket/resolve-merge-conflicts-704414003.html)。

一旦一切顺利完成，您的更新将最终与中央存储库合并到`master`分支中。恭喜你！

# 最后的想法

![](img/cdf54d31d11efd975ea1c3c61ef70922.png)

[(Source)](https://unsplash.com/photos/NXxhXWFEZ6E)

感谢您的阅读。

当我第一次开始学习 Git 时，我感到非常沮丧，因为我仍然没有真正理解工作流——大图——尽管我理解了一般的术语。

这是我写这篇文章的主要原因之一，以便在更高的理解水平上对工作流进行真正的分解和解释。因为我相信对工作流程中发生的事情有一个清晰的理解会使学习过程更有效。

希望这篇分享在某些方面对你有益。

一如既往，如果您有任何问题或意见，请随时在下面留下您的反馈，或者您可以随时通过 [LinkedIn](https://www.linkedin.com/in/admond1994/) 联系我。在那之前，下一篇文章再见！😄

## 关于作者

[**阿德蒙德·李**](https://www.linkedin.com/in/admond1994/) 目前是东南亚排名第一的商业银行 API 平台 [**Staq**](https://www.trystaq.com) **—** 的联合创始人/首席技术官。

想要获得免费的每周数据科学和创业见解吗？

你可以在 [LinkedIn](https://www.linkedin.com/in/admond1994/) 、 [Medium](https://medium.com/@admond1994) 、 [Twitter](https://twitter.com/admond1994) 、[脸书](https://www.facebook.com/admond1994)上和他联系。

[](https://www.admondlee.com/) [## 阿德蒙德·李

### 让每个人都能接触到数据科学。Admond 正在通过先进的社交分析和机器学习，利用可操作的见解帮助公司和数字营销机构实现营销投资回报。

www.admondlee.com](https://www.admondlee.com/)