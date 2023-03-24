# 使用 Kaggle 和 GitHub 操作的可再现数据科学

> 原文：<https://towardsdatascience.com/reproducible-data-science-using-kaggle-and-github-actions-b0d78380bf8e?source=collection_archive---------22----------------------->

## 本教程演示了如何将 Kaggle 与 GitHub 操作集成在一起，以便更好地再现数据科学项目。

![](img/6fe7bc50a3fbf92324279b43733a2e73.png)

Reproducibility means hitting the right target, every time (Photo by [Oliver Buchmann](https://unsplash.com/@snxiiy?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/bullseye?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText))

随着数据科学中出现的[再现性危机，对数据科学研究人员来说，提供对其代码的开放访问变得越来越重要。其中一个基本要素是确保现有实验在代码变化时的持续性能。通过使用测试和记录良好的代码，可以提高正在进行的项目的可重复性。](/data-sciences-reproducibility-crisis-b87792d88513)

# Kaggle 和 GitHub 操作

[Kaggle](https://www.kaggle.com/) 是最知名的数据科学社区之一，致力于寻找数据集、与其他数据科学家合作以及参加竞赛。对于所有级别的从业者来说，这是一个极好的资源，并且提供了一个[强大的 API](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication) 来访问它的资源。通过使用这个 API，可以创建自动下载数据集的测试，算法可以根据这些数据集运行，确保算法在代码更新时继续按预期执行。

GitHub 最近发布了 [GitHub Actions](https://github.com/features/actions) ，这是一个直接从 GitHub 仓库自动化工作流的集成平台。本文的其余部分演示了如何将 GitHub 动作与 Kaggle API 一起使用，以允许对数据科学应用程序进行连续测试。使用 Kaggle API 的 GitHub 动作的完整实现可以在[这里](https://github.com/JAEarly/KaggleActionsExample)找到。

# 工作流程分解

下面详细介绍了将 Kaggle 与 GitHub 动作集成的最基本设置。这不包括数据科学测试脚本的实际执行，因为它们依赖于语言。相反，它展示了如何在 GitHub 动作中运行 Kaggle API 命令，到达可以执行测试脚本的地方。

工作流程的基本要素如下:

1.  在测试环境中设置 Python
2.  安装 Kaggle Python API
3.  执行 Kaggle 命令
4.  运行测试脚本(省略)

# 履行

GitHub 操作是使用 YAML 文件创建作业并指定作业中的步骤来实现的。以下作业中的每个步骤都与上面工作流中的元素相对应。

```
- name: Setup python
  uses: actions/setup-python@v1
  with:
    python-version: 3.6
    architecture: x64
- name: Setup Kaggle
  run: pip install kaggle
- name: Run Kaggle command
  run: kaggle competitions list 
  env:
    KAGGLE_USERNAME: ${{ secrets.KaggleUsername }}
    KAGGLE_KEY: ${{ secrets.KaggleKey }}
```

这个实现利用了 Kaggle PyPi 包，该包允许 Kaggle API 命令通过命令行运行。在这个例子中，job 简单地列出了 Kaggle 上可用的竞争对手。可用 Kaggle API 命令的分类可以在[这里](https://github.com/Kaggle/kaggle-api)找到。

也可以像 GitHub actions 中的其他任务一样，将多个 Kaggle 命令链接在一起。在这里，工作流步骤显示 Kaggle 版本，然后列出所有可用的竞赛。

```
- name: Run multiple Kaggle commands
      run: |
        kaggle --version
        kaggle competitions list
      env:
        KAGGLE_USERNAME: ${{ secrets.KaggleUsername }}
        KAGGLE_KEY: ${{ secrets.KaggleKey }}
```

# 向 Kaggle 认证

使用 Kaggle API 需要一个访问令牌。这些可以在 Kaggle 网站上[创建。令牌通常通过文件提供给 Kaggle API，但是 KAGGLE_USERNAME 和 KAGGLE_KEY 环境变量也可以用来进行身份验证。可以在 GitHub actions](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication) 中配置环境变量[，如上面代码中的 env 参数所示。令牌变量被保存为](https://help.github.com/en/actions/automating-your-workflow-with-github-actions/using-environment-variables) [GitHub secrets](https://help.github.com/en/actions/automating-your-workflow-with-github-actions/creating-and-using-encrypted-secrets) 以确保个人令牌不被暴露。

这段代码的完整实现可以在[这里](https://github.com/JAEarly/KaggleActionsExample)找到。希望本指南为您提供了在 GitHub actions 中使用 Kaggle API 的快速介绍，并允许您开始更彻底地测试您的数据科学代码！