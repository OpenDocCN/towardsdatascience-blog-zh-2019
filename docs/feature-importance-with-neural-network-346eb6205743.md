# 神经网络的特征重要性

> 原文：<https://towardsdatascience.com/feature-importance-with-neural-network-346eb6205743?source=collection_archive---------1----------------------->

## 让机器学习变得容易理解，提供变量关系解释

![](img/533e4795ac7508d32ac916f7da54b062.png)

Photo by [Markus Spiske](https://unsplash.com/@markusspiske?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

机器学习中最大的挑战之一是让模型自己说话。不仅开发具有强大预测能力的强大解决方案很重要，而且在许多商业应用中，了解模型如何提供这些结果也很有趣:哪些变量参与最多，相关性的存在，可能的因果关系等等。

这些需求使得基于树的模型成为这个领域的一个好武器。它们是可扩展的，并且允许非常容易地计算变量解释。每个软件都提供这个选项，我们每个人都至少尝试过一次用随机森林或类似的方法计算变量重要性报告。对于神经网络，这种益处被认为是禁忌。神经网络通常被视为一个黑箱，很难从其中提取有用的信息用于其他目的，如特征解释。

在这篇文章中，我试图提供一个优雅而聪明的解决方案，用几行代码，允许你挤压你的机器学习模型并提取尽可能多的信息，以便**提供特征重要性，个性化重要的相关性并试图解释因果关系**。

# 数据集

给定一个真实的数据集，我们试图研究哪些因素影响最终的预测性能。为了实现这个目标，我们从 UCI 机器学习知识库中获取数据。特权数据集是[联合循环发电厂](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant)数据集，当发电厂设置为满负荷工作时，在那里收集了 *6 年的数据。特征包括每小时平均变量:环境温度(AT)、环境压力(AP)、相对湿度(RH)和排气真空(V ),以预测电厂的每小时净电能输出(PE)。*

所涉及的变量通过 Pearson 相关链接进行关联，如下表所示。

![](img/e580e64261792e8c7d876c98d9ef5c0f.png)

Correlation Matrix

# 梯度增强特征重要性

我们开始构建一个简单的基于树的模型，以便提供能量输出 *(PE)* 预测，并计算标准的特征重要性估计。这最后一步使我们能够比标准的相关指数更多地说明变量之间的关系。这些数字总结了在内部空间划分期间(在训练阶段)指出特定特征时，所有树的杂质指数的减少。Sklearn 应用归一化，以便提供可累加为 1 的输出。这也是一个免费的结果，可以在训练后间接获得。

```
gb = GradientBoostingRegressor(n_estimators=100)
gb.fit(X_train, y_train.values.ravel())plt.bar(range(X_train.shape[1]), gb.feature_importances_)
plt.xticks(range(X_train.shape[1]), ['AT','V','AP','RH'])
```

![](img/e2ec0cbc09a821f59496cee233b04827.png)

GradientBoosting Features Importance

这个结果很容易解释，并且似乎复制了计算与我们的目标变量的相关性的初始假设(相关矩阵的最后一行):值越高，这个特定特征预测我们的目标的影响就越大。

尽管我们通过梯度推进取得了很好的结果，但我们不想完全依赖这种方法……我们想推广计算特征重要性的过程，让我们自由开发另一种具有相同灵活性和解释能力的机器学习模型；更进一步:提供变量之间存在显著伤亡关系的证据。

# 排列重要性

为我们的实验确定的模型无疑是神经网络，因为它们享有黑盒算法的声誉。为了揭开这个刻板印象的神秘面纱，我们将关注排列的重要性。其易于实现，结合其有形的理解和适应性，使其成为回答问题的一致候选:*哪些特性对预测的影响最大？*

排列重要性**是在模型拟合后计算的。所以我们只能压榨它，得到我们想要的。这种方法的工作原理很简单:*如果我随机打乱数据中的单个特征，让目标和所有其他特征保持不变，这会如何影响最终的预测性能*？**

从变量的随机重新排序中，我期望得到:

*   不太准确的预测，因为产生的数据不再符合现实世界中观察到的任何东西；
*   最差的表现，来自最重要变量的混乱。这是因为我们正在破坏数据的自然结构。如果我们的洗牌破坏了一个牢固的关系，我们将损害我们的模型在训练中所学到的东西，导致更高的错误(**高错误=高重要性**)。

![](img/65cf0f437c3404b0afb96113e985e4ae.png)

Permutation Importance at work

实际上，这是我们真实场景中发生的事情…

我们选择了合适的神经网络结构来模拟*小时电能输出* ( *EP* )。记住也要在一个较低的范围内调整目标变量:我经典地减去平均值，除以标准差，这有助于训练。

```
inp = Input(shape=(scaled_train.shape[1],))
x = Dense(128, activation='relu')(inp)
x = Dense(32, activation='relu')(x)
out = Dense(1)(x)model = Model(inp, out)
model.compile(optimizer='adam', loss='mse')model.fit(scaled_train, (y_train - y_train.mean())/y_train.std() , epochs=100, batch_size=128 ,verbose=2)
```

在预测阶段，梯度提升和神经网络在平均绝对误差方面达到相同的性能，分别为 2.92 和 2.90(记住要反向预测)。

至此，我们结束了培训，让我们开始随机抽样。

我们计算验证数据上每个特征的混洗(总共 4 次= 4 个显式变量),并提供每一步的误差估计；记住每一步都要把数据恢复到原来的顺序。然后，我将我们在每个洗牌阶段获得的 MAE 绘制成相对于原始 MAE 的百分比变化(大约 2.90)

```
plt.bar(range(X_train.shape[1]), (final_score - MAE)/MAE*100)
plt.xticks(range(X_train.shape[1]), ['AT','V','AP','RH'])
```

![](img/d0f22bdb60efc9d2d45867c94f9f5feb.png)

Permutation Importance as percentage variation of MAE

上图复制了射频特征重要性报告，并证实了我们最初的假设:环境温度(AT)*是预测电能输出(PE)* 的最重要和最相关的特征。尽管处的*排气真空(V)* 和*与 *PE* (分别为 0.87 和 0.95)表现出相似且高度相关的关系，但它们在预测阶段具有不同的影响。这一现象是一个软例子，说明高相关性(皮尔逊术语)并不总是高解释力的同义词。*

# 因果关系

为了避免[虚假关系](https://www.tylervigen.com/spurious-correlations)，证明相关性总是一种阴险的操作。同时，很难出示伤亡行为的证据。在文献中，有很多证明因果关系的方法。其中最重要的是[格兰杰因果关系检验](https://en.wikipedia.org/wiki/Granger_causality)。这种技术广泛应用于时间序列领域，以确定一个时间序列是否有助于预测另一个时间序列:即证明(根据滞后值的 f 检验)它增加了回归的解释力。

间接的，这就是我们已经做的，计算排列重要性。改变每一个变量并寻找性能变化，我们正在证明这个特性在预测预期目标方面有多大的解释力。

为了证明因果关系，我们现在要做的是证明数据混洗为性能变化提供了有意义的证据。我们对无洗牌和有洗牌的最终预测进行操作，并验证两个预测总体的平均值是否有差异。这意味着随机的平均预测也可以被任何随机的预测子群观察到。因此，这正是我们将对每个特征所做的:我们将合并有置换和无置换的预测，我们将随机抽样一组预测，并计算它们的平均值与无置换预测的平均值之间的差异。

```
np.random.seed(33)
id_ = 0 #feature index
merge_pred = np.hstack([shuff_pred[id_], real_pred])
observed_diff = abs(shuff_pred[id_].mean() - merge_pred.mean())
extreme_values = []
sample_d = []for _ in range(10000):
    sample_mean = np.random.choice(merge_pred,
                  size=shuff_pred[id_].shape[0]).mean()
    sample_diff = abs(sample_mean - merge_pred.mean())
    sample_d.append(sample_diff)
    extreme_values.append(sample_diff >= observed_diff)

np.sum(extreme_values)/10000 #p-value
```

为了控制一切，可视化我们的模拟结果是一个很好的选择。我们绘制了模拟平均差异的分布(蓝色条)并标记了实际观察到的差异(红线)。我们可以看到，对于*在*处，有证据表明在没有滑移的情况下做出的预测在均值上存在差异(低 p 值:低于 0.1)。其他变量不会带来均值的显著提高。

![](img/8614e23e2a979de3b2d84fc6b3c9f07a.png)

Simulation distributions and relative p-values

相关性并不总是意味着因果关系！考虑到这一点，**我们根据一个选定特征增加解释力的能力来证明因果关系**。我们用我们的统计学家和程序员的知识，重新创造了一种方法来证明这个概念，利用了我们以前在排列重要性方面的发现，增加了关于变量关系的信息。

# 摘要

在这篇文章中，我介绍了排列重要性，这是一种计算特性重要性的简单而聪明的技术。它对各种模型(我使用神经网络只是作为个人选择)和每个问题都很有用(模拟程序适用于分类任务中的**:在计算排列重要性时，记得选择适当的损失度量**，如交叉熵，避免模糊的准确性)。我们还使用了排列来展示一种方法，证明了攻击 p 值的变量之间的因果关系！

[**查看我的 GITHUB 回购**](https://github.com/cerlymarco/MEDIUM_NoteBook)

保持联系: [Linkedin](https://www.linkedin.com/in/marco-cerliani-b0bba714b/)