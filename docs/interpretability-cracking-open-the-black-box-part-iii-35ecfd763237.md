# 可解释性:打开黑盒——第三部分

> 原文：<https://towardsdatascience.com/interpretability-cracking-open-the-black-box-part-iii-35ecfd763237?source=collection_archive---------43----------------------->

## 深入了解石灰、匀称的价值观和 SHAP

付费墙是否困扰着你？点击 [*这里*](/interpretability-cracking-open-the-black-box-part-iii-35ecfd763237?source=friends_link&sk=feefa4d113a26383994c93ce3db41f41) *可以绕过去。*

之前，我们[在基于树的模型中使用默认的*特征重要性*查看了陷阱](https://deep-and-shallow.com/2019/11/16/interpretability-cracking-open-the-black-box-part-ii/)，讨论了排列重要性、LOOC 重要性和部分依赖图。现在让我们换个话题，看看一些模型不可知的技术，它们采用自下而上的方式来解释预测。这些方法不是着眼于模型并试图给出像特征重要性这样的全局解释，而是着眼于每一个预测，然后试图解释它们。

![](img/d0b20e915d4451f6018a22181a7b64ed.png)

# 5.局部可解释的模型不可知解释(LIME)

顾名思义，这是一种模型不可知的技术，为模型生成局部解释。这项技术背后的核心思想非常直观。假设我们有一个复杂的分类器，具有高度非线性的决策边界。但是，如果我们放大观察一个单一的预测，该模型在该地区的行为可以用一个简单的可解释模型(大部分是线性的)来解释。

![](img/8414b77a6a4d68d08051a84b5755ab9e.png)

LIME[2]使用了一个局部替代模型，该模型针对我们正在研究的数据点的扰动进行训练，以寻求解释。这确保了即使解释不具有全局保真度(对原始模型的保真度),它也具有局部保真度。论文[2]还认识到可解释性与保真度之间存在权衡，并提出了一个表达该框架的正式框架。

![](img/d6bb4fe86c90d33f7929f3d2d62e9a6f.png)

ξ *(x)* 是解释，L(f，g，πₓ)是局部保真度的倒数(或者说 *g* 在局部逼近 *f* 有多不忠实)，ω*(g)*是局部模型的复杂度， *g* 。为了确保局部保真度和可解释性，我们需要最小化不忠实度(或最大化局部保真度)，牢记复杂度应该低到足以让人类理解。

即使我们可以使用任何可解释的模型作为局部替代，本文使用套索回归来诱导解释的稀疏性。该论文的作者将他们的探索限制在模型的保真度上，并将复杂性保持为用户输入。在套索回归的情况下，它是解释应归因于的要素的数量。

他们探索并提出解决方案的另一个方面(还没有得到很多人的欢迎)是使用一组个体实例提供全局解释的挑战。他们称之为*“子模式摘”。它本质上是一种贪婪的优化，试图从整个批次中挑选一些实例，最大化他们所谓的“非冗余覆盖”。非冗余覆盖确保优化不会挑选具有相似解释的实例。*

*该技术的**优势**是:*

*   *方法论和解释对人类来说都是非常直观的。*
*   *生成的解释是稀疏的，从而增加了可解释性。*
*   *模型不可知*
*   *LIME 适用于结构化和非结构化数据(文本、图像)*
*   *在 [R](https://github.com/thomasp85/lime) 和 [Python](https://github.com/marcotcr/lime) 中随时可用(本文作者的原始实现)*
*   *使用其他可解释特征的能力，即使模型是在嵌入等复杂特征中训练的。例如，回归模型可以在 PCA 的几个成分上训练，但是解释可以在对人类有意义的原始特征上生成。*

## *算法*

*   *为了找到对单个数据点和给定分类器的解释*
*   *均匀随机地对所选单个数据点周围的局部性进行采样，并从我们想要解释的模型中生成扰动数据点及其相应预测的数据集*
*   *使用指定的特性选择方法选择解释所需的特性数量*
*   *使用核函数和距离函数计算样本权重。(这捕获了采样点离原始点有多近或多远)*
*   *使用样本权重对目标函数进行加权，在扰动的数据集上拟合可解释的模型。*
*   *使用新训练的可解释模型提供本地解释*

## *履行*

*论文作者的实现可以在 Github 和 pip 中找到。在我们看一下如何实现它们之前，让我们先讨论一下实现中的一些奇怪之处，在运行它之前你应该知道这些(重点是表格解释器)。*

*主要步骤如下*

*   *通过提供训练数据(或训练数据统计，如果训练数据不可用)，一些关于特性和类名的细节(在分类的情况下)，初始化*tabular 解释器**
*   *调用类中的方法， *explain_instance* 并提供需要解释的实例、训练模型的预测方法以及需要包含在解释中的功能数量。*

*你需要记住的关键事情是:*

*   *默认情况下*模式*是分类。所以如果你试图解释一个回归问题，一定要提到它。*
*   *默认情况下，功能选择设置为“自动”。“自动”基于训练数据中的特征数量在 [*前向选择*](https://www.kdnuggets.com/2018/06/step-forward-feature-selection-python.html) 和*最高权重*之间选择(如果小于 6，前向选择。最大权重刚好符合对缩放数据的岭回归，并选择 n 个最大权重*和*。如果将“无”指定为特征选择参数，则不会进行任何特征选择。如果您将“lasso_path”作为特征选择，它将使用来自 sklearn 的信息来查找提供 *n* 个非零特征的正确级别正则化。*
*   *默认情况下，岭回归用作可解释模型。但是您可以传入任何您想要的 sci-kit 学习模型，只要它有 coef_ 和' sample_weight '作为参数。*
*   *还有另外两个关键参数， *kernel* 和 *kernel_width* ，它们决定了样本权重的计算方式，也限制了可能发生扰动的局部性。这些在算法中作为超参数保存。虽然，在大多数用例中，缺省值是可行的。*
*   *默认情况下，*discrete ize _ continuous*设置为 *True* 。这意味着连续特征被离散为四分位数、十分位数或基于熵。*离散化器*的默认值为*四分位数*。*

*现在，让我们继续使用上一部分中使用的数据集，看看 LIME 的实际应用。*

```
*import lime import lime.lime_tabular 
# Creating the Lime Explainer 
# Be very careful in setting the order of the class names lime_explainer = lime.lime_tabular.LimeTabularExplainer( X_train.values, training_labels=y_train.values, feature_names=X_train.columns.tolist(), feature_selection="lasso_path", class_names=["<50k", ">50k"], discretize_continuous=True, discretizer="entropy", )
#Now let's pick a sample case from our test set. 
row = 345*
```

*![](img/63787217e979cb5f3512b5240a835279.png)*

*Row 345*

```
*exp = lime_explainer.explain_instance(X_test.iloc[row], rf.predict_proba, num_features=5) exp.show_in_notebook(show_table=True)*
```

*![](img/d7a1be8740922b381e32058c5b731ff8.png)*

*为了多样化，让我们看另一个例子。一个被模型错误分类的。*

*![](img/14d7b3efcb922e128b5762d9b6d0ad4a.png)*

*Row 26*

*![](img/ec58d7efdff7d6cef43df0a1d93c9fe8.png)*

## *解释*

**例 1**

*   *我们看到的第一个例子是一个 50 岁的已婚男子，拥有学士学位。他在一家私营公司担任行政/管理职务，每周工作 40 个小时。我们的模型正确地将他归类为收入超过 5 万英镑的人*
*   *这样的预测在我们的心理模型中是有意义的。一个 50 岁的人在管理岗位上工作，收入超过 50k 的可能性非常高。*
*   *如果我们看看这个模型是如何做出这个决定的，我们可以看到，这是由于他的婚姻状况、年龄、教育程度以及他担任行政/管理职务的事实。事实上，他的职业不是一个专业，这试图降低他的可能性，但总的来说，模型决定，这个人很有可能收入超过 50k*

**例 2**

*   *第二个例子是一个 57 岁的已婚男子，高中学历。他是一名从事销售的个体户，每周工作 30 个小时。*
*   *即使在我们的心理模型中，我们也很难预测这样一个人的收入是否超过 50k。模型预测这个人的收入不到 50k，而事实上他赚得更多。*
*   *如果我们看看模型是如何做出这个决定的，我们可以看到在预测的局部有一个很强的推拉效应。一方面，他的婚姻状况和年龄将他推向了 50k 以上。另一方面，他不是高管/经理的事实、他所受的教育和每周工作时间都在把他往下推。最终，向下推动赢得了比赛，模型预测他的收入低于 50k。*

## *子模块选择和全局解释*

*如前所述，论文中还提到了另一种技术，叫做“子模型选择”,可以找到一些解释来解释大多数情况。让我们也试着得到它。python 库的这个特殊部分不太稳定，提供的示例笔记本给了我错误。但是在花了一些时间通读源代码之后，我找到了一个解决错误的方法。*

```
*from lime import submodular_pick 
sp_obj = submodular_pick.SubmodularPick(lime_explainer, X_train.values, rf.predict_proba, sample_size=500, num_features=10, num_exps_desired=5) #Plot the 5 explanations [exp.as_pyplot_figure(label=exp.available_labels()[0]) for exp in sp_obj.sp_explanations]; # Make it into a dataframe W_pick=pd.DataFrame([dict(this.as_list(this.available_labels()[0])) for this in sp_obj.sp_explanations]).fillna(0) 
W_pick['prediction'] = [this.available_labels()[0] for this in sp_obj.sp_explanations] #Making a dataframe of all the explanations of sampled points W=pd.DataFrame([dict(this.as_list(this.available_labels()[0])) for this in sp_obj.explanations]).fillna(0) 
W['prediction'] = [this.available_labels()[0] for this in sp_obj.explanations] #Plotting the aggregate importances 
np.abs(W.drop("prediction", axis=1)).mean(axis=0).sort_values(ascending=False).head( 25 ).sort_values(ascending=True).iplot(kind="barh") #Aggregate importances split by classes 
grped_coeff = W.groupby("prediction").mean() 
grped_coeff = grped_coeff.T 
grped_coeff["abs"] = np.abs(grped_coeff.iloc[:, 0]) grped_coeff.sort_values("abs", inplace=True, ascending=False) grped_coeff.head(25).sort_values("abs", ascending=True).drop("abs", axis=1).iplot( kind="barh", bargap=0.5 )*
```

*![](img/6f937a0a3ca1cacff30fa1898fe9b98a.png)*

*[Click for full interactive plot](https://plot.ly/~manujosephv/43/)*

*![](img/8ec883e1eadc588cb8d8a6f6ee0d25b0.png)*

*[Click for full interactive plot](https://plot.ly/~manujosephv/49/)*

## *解释*

*有两个图表，其中我们汇总了从我们的测试集中采样的 500 个点的解释(我们可以在所有测试数据点上运行它，但选择只进行采样，因为需要计算)。*

*第一个图表汇总了超过 50k 和<50k cases and ignores the sign when calculating the mean. This gives you an idea of what features were important in the larger sense.*

*The second chart splits the inference across the two labels and looks at them separately. This chart lets us understand which feature was more important in predicting a particular class.*

*   *Right at the top of the first chart, we can find “*婚姻状况> 0.5* 的特征的影响。按照我们的编码，就是单身的意思。所以单身是一个很强的指标，表明你的收入是高于还是低于 50k。但是等等，结婚是第二位的。这对我们有什么帮助？*
*   *如果你看第二张图表，情况会更清楚。你可以立即看到单身让你进入了<50k bucket and being married towards the >的 50k 桶。*
*   *在你匆忙去寻找伴侣之前，请记住这是模型用来寻找伴侣的。这不一定是现实世界中的因果关系。也许这个模型是在利用其他一些与结婚有很大关联的特征来预测收入潜力。*
*   *我们在这里也可以看到性别歧视的痕迹。如果你看“性别> 0.5”，这是男性，两个收入潜力阶层之间的分布几乎相等。不过只要看一下《性<0.5”. It shows a large skew towards <50k bucket.*

*Along with these, the submodular pick also(in fact this is the main purpose of the module) a set of n data points from the dataset, which best explains the model. We can look at it like a representative sample of the different cases in the dataset. So if we need to explain a few cases from the model to someone, this gives you those cases which will cover the most ground.*

## *The Joker in the Pack*

*From the looks of it, this looks like a very good technique, isn’t it? But it is not without its problems.*

*The biggest problem here is the correct definition of the neighbourhood, especially in tabular data. For images and text, it is more straightforward. Since the authors of the paper left kernel width as a hyperparameter, choosing the right one is left to the user. But how do you tune the parameter when you don’t have a ground truth? You’ll just have to try different widths, look at the explanations, and see if it makes sense. Tweak them again. But at what point are we crossing the line into tweaking the parameters to get the explanations we want?*

*Another main problem is similar to the problem we have with permutation importance( [第二部](https://deep-and-shallow.com/2019/11/16/interpretability-cracking-open-the-black-box-part-ii/))就知道了。当对局部中的点进行采样时，LIME 的当前实现使用高斯分布，这忽略了特征之间的关系。这可以创建相同的*‘不太可能’*数据点，在这些数据点上学习解释。*

*最后，选择一个线性的可解释模型来进行局部解释可能并不适用于所有情况。如果决策边界过于非线性，线性模型可能无法很好地解释它(局部保真度可能较高)。*

# *6.得体的价值观*

*在讨论 Shapely 值如何用于机器学习模型解释之前，我们先试着了解一下它们是什么。为此，我们必须简单地了解一下博弈论。*

*博弈论是数学中最迷人的分支之一，它研究理性决策者**之间**战略互动**的数学模型**。当我们说游戏时，我们指的不仅仅是象棋，或者就此而言，垄断。游戏可以概括为两个或两个以上的玩家/团体参与一个决策或一系列决策以改善他们的地位的任何情况。当你这样看待它时，它的应用扩展到战争策略、经济策略、扑克游戏、定价策略、谈判和合同，不胜枚举。*

*但是由于我们关注的主题不是博弈论，我们将只讨论一些主要术语，这样你就能跟上讨论。参与**游戏**的各方被称为**玩家**。这些玩家可以采取的不同行动被称为**选择**。如果每个参与者都有有限的选择，那么每个参与者的选择组合也是有限的。因此，如果每个玩家都做出一个选择，就会产生一个**结果**，如果我们量化这些结果，这就叫做**收益**。如果我们列出所有的组合和与之相关的收益，这就叫做**收益矩阵**。*

*博弈论有两种范式——非合作、合作博弈。而 Shapely 值是合作博弈中的一个重要概念。我们试着通过一个例子来理解。*

*爱丽丝、鲍伯和席琳一起用餐。账单共计 90 英镑，但他们不想各付各的。因此，为了算出他们各自欠了多少钱，他们以不同的组合多次去同一家餐馆，并记录下账单金额。*

*![](img/5eb630ea4b977d0d7f62ad61036fde27.png)*

*有了这些信息，我们做一个小的心理实验。假设 A 去了餐厅，然后 B 出现了，C 出现了。因此，对于每个加入的人，我们可以有每个人必须投入的额外现金(边际贡献)。我们从 80 英镑开始(如果 A 一个人吃饭，他会支付 80 英镑)。现在当 B 加入时，我们看看 A 和 B 一起吃饭时的收益——也是 80。所以 B 给联盟带来的额外贡献是 0。当 C 加入时，总收益是 90。这使得 C 10 的边际贡献。所以，当 A，B，C 依次加入时的贡献是(80，0，10)。现在我们对这三个朋友的所有组合重复这个实验。*

*现在我们有了所有可能的到达顺序，我们有了所有参与者在所有情况下的边际贡献。每个参与者的预期边际贡献是他们在所有组合中边际贡献的平均值。例如，A 的边际贡献将是，*(80+80+56+16+5+70)/6 = 51.17*。如果我们计算每个玩家的预期边际贡献，并把它们加在一起，我们会得到 90，这是三个人一起吃饭的总收益。*

*你一定想知道所有这些与机器学习和可解释性有什么关系。很多。如果我们想一想，一个机器学习预测就像一个**游戏**，其中不同的功能(**玩家**)，一起游戏以带来一个结果(**预测**)。由于这些特征一起工作，相互作用，做出预测，这就变成了一个合作博弈的例子。这完全符合匀称的价值观。*

*但是只有一个问题。随着特征的增加，计算所有可能的联盟及其结果很快变得不可行。因此，2013 年，Erik trumbelj*等人*提出了一种使用蒙特卡罗抽样的近似方法。在这个结构中，回报被建模为不同蒙特卡罗样本的预测与平均预测的差异。*

*![](img/948c7a66b429fd43a2b8afe3a5ca6b7e.png)*

*其中 f 是我们试图解释的黑盒机器学习模型， *x* 是我们试图解释的实例，j 是我们试图找到预期边际贡献的特征， *x* ᵐ₋ⱼ和 *x* ᵐ₊ⱼ是 *x* 的两个实例，我们已经通过从数据集本身中采样另一个点对其进行了随机置换， *M* 是我们从训练集中抽取的样本数。*

*让我们看看 Shapely 值的一些令人满意的数学属性，这使得它在可解释性应用中非常令人满意。Shapely 值是唯一满足效率、对称性、虚拟性和可加性属性的属性方法。同时满足这些条件被认为是公平支付的定义。*

*   *效率-要素贡献加起来就是 x 和平均值的预测差异。*
*   *对称性-如果两个特征值对所有可能的联合的贡献相等，则它们的贡献应该相同。*
*   *虚拟-无论添加到哪个联盟，都不会更改预测值的要素的 Shapely 值都应为 0*
*   *可加性——对于组合支付的游戏，可以将相应的 Shapely 值加在一起，得到最终的 Shapely 值*

*虽然所有的属性都使这成为一种理想的特征归属方式，但其中一个特别具有深远的影响，即*可加性*。这意味着，对于像 RandomForest 或梯度增强这样的集合模型，该属性保证了如果我们单独计算每棵树的要素的 Shapely 值并对其进行平均，您将获得集合的 Shapely 值。这一特性也可以扩展到其他集成技术，如模型叠加或模型平均。*

*出于两个原因，我们将不回顾算法并查看 Shapely 值的实现:*

*   *在大多数现实世界的应用中，计算 Shapely 值是不可行的，即使有近似值。*
*   *有一个更好的计算 Shapely 值的方法，还有一个稳定的库，我们将在下面介绍。*

# *7.得体的附加解释(SHAP)*

*SHAP (SHapely 附加解释)提出了一个解释模型预测的统一方法。斯科特·伦德伯格*等人*提出了一个框架，该框架统一了六种先前存在的特征归属方法(包括 LIME 和 DeepLIFT ),他们将其框架作为附加特征归属模型。*

*![](img/2607d96f9302ad164e2cbe71ae6bf9df.png)*

*他们表明，这些方法中的每一种都可以用上面的等式来表示，并且 Shapely 值可以很容易地计算出来，这带来了一些保证。尽管论文中提到的属性与 Shapely 值略有不同，但原则上它们是相同的。这提供了一个强大的技术(如石灰)的理论基础，一旦适应这一框架的估计匀称的价值。在论文中，作者提出了一种新的模型不可知的方法来近似 Shapely 值，称为核 SHAP(石灰+ Shapely 值)和一些特定于模型的方法，如 DeepSHAP(这是 DeepLIFT 的改编，一种估计神经网络特征重要性的方法)。除此之外，他们还表明，对于线性模型，如果我们假设特征独立，Shapely 值可以直接从模型的权重系数中近似得到。2018 年，斯科特·伦德伯格*等人*【6】提出了该框架的另一个扩展，该框架可以精确计算基于树的系综的 Shapely 值，如随机森林或梯度增强。*

# *内核 SHAP*

*尽管从下面的等式来看，这并不是非常直观，但石灰也是一种附加的特征归属方法。对于附加特征解释方法，Scott Lundberg *等人*表明，满足所需属性的唯一解决方案是 Shapely 值。并且该解取决于损失函数 *L* 、加权核πₓ和正则化项ω。*

*如果你还记得，当我们讨论 LIME 时，我提到过它的一个缺点是，它将核函数和核距离作为超参数，它们是使用启发式算法来选择的。内核 SHAP 消除了这种不确定性，提出了一个形状良好的内核和相应的损失函数，确保上述方程的解决方案将产生形状良好的值，并享受数学保证。*

## *算法*

*   *对联合向量进行采样(联合向量是二进制值的向量，其长度与特征的数量相同，表示特定特征是否包括在联合中。 *z'* ₖ ϵ {0,1}ₘ， *k* ϵ { *1* ，…， *K* } (1 =特征存在，0 =特征不存在)*
*   *通过使用函数 *hₓ* 将联合向量转换到原始样本空间，从模型中获得联合向量 *z'ₖ* 的预测。 *hₓ* 只是我们用一套变换从原始输入数据中得到相应值的一种花哨说法。例如，对于联合向量中的所有 1，我们用我们正在解释的实例中该特征的实际值来替换它。对于 0 来说，它根据应用略有不同。对于表格数据，从数据中随机抽样，用相同特征的一些其他值替换 0。对于图像数据，0 可以用参考值或零像素值代替。下图试图让表格数据的这一过程变得清晰。*
*   *使用 Shapely 内核计算样本的重量*
*   *对 K 个样本重复此操作*
*   *现在，拟合一个加权线性模型并返回 Shapely 值，即模型的系数。*

*![](img/bc58faa20ef20ae7c1bf9ed00569a675.png)*

# *特里 SHAP*

*如前所述[6]，树 SHAP 是一种快速算法，它为基于决策树的模型计算精确的 Shapely 值。相比之下，核 SHAP 只是近似 Shapely 值，计算起来要昂贵得多。*

## *算法*

*让我们试着直观地了解一下它是如何计算的，而不要深入研究大量的数学知识(那些倾向于数学的人，这篇论文在参考资料中有链接，尽情享受吧！).*

*我们将首先讨论算法如何对单棵树起作用。如果你记得关于 Shapely 值的讨论，你会记得为了精确计算，我们需要以实例的特征向量的所有子集为条件的预测。因此，假设我们试图解释的实例的特征向量是 *x* ，并且我们需要预期预测的特征子集是 *S* 。*

*下面是一个人工决策树，它仅使用三个特征*年龄、工作时间和婚姻状况*来预测收入潜力。*

*![](img/6cb2492b92152fb7190176c52405a438.png)*

*   *如果我们以所有特征为条件，即 *S* 是所有特征的集合，那么 *x* 所在的节点中的预测就是预期预测。即> 50k*
*   *如果我们不以任何特征为条件，则同样有可能(忽略点在节点间的分布)您最终会出现在任何决策节点中。因此，预期预测将是所有终端节点预测的加权平均值。在我们的例子中，有 3 个节点输出 1( <50k) and three nodes which outputs 0 (> 50k)。如果我们假设所有的训练数据点平均分布在这些节点上，在没有所有特征的情况下的预期预测是 0.5*
*   *如果我们以某些特性 *S* 为条件，我们计算出实例 x 最终可能到达的节点的期望值。例如，如果我们从集合 *S* 中排除*marriage _ status*，实例同样有可能在节点 5 或节点 6 中结束。因此，对于这种 *S* 的预期预测将是节点 5 和 6 的输出的加权平均值。因此，如果我们从 *S* 中排除 *hours_worked* ，预期预测会有任何变化吗？否，因为 hours_worked 不在实例 *x* 的决策路径中。*
*   *如果我们排除了位于树根的特征，如*年龄*，它将创建多个子树。在这种情况下，它将有两棵树，一棵树从右侧的*已婚*区块开始，另一棵树从左侧的*工作小时数*开始。还有一个节点 4 的决策存根。现在实例 *x* 沿着两棵树向下传播(不包括具有*年龄*的决策路径)，并且预期预测被计算为所有可能节点(节点 3、4 和 5)的加权平均值。*

*现在，您已经在一个决策树中获得了所有子集的预期预测，您可以对集合中的所有树重复该操作。还记得 Shapely 值的*可加性*属性吗？它允许您通过计算所有树的 Shapely 值的平均值，将它们聚集在一个系综中。*

*但是，现在的问题是，必须为所有树中所有可能的特征子集和所有特征计算这些期望值。该论文的作者提出了一种算法，在这种算法中，我们能够同时将所有可能的特征子集推下树。这个算法相当复杂，我建议你参考参考文献中链接的论文来了解细节。*

## *优势*

*   *SHAP 和 Shapely 价值观享有博弈论的坚实理论基础。Shapely 值保证了预测在不同的要素之间公平分布。这可能是唯一能够经受住理论和实践检验冲击的特征归属技术，无论是学术检验还是监管检验*
*   *SHAP 将其他可解释性技术，如 LIME 和 DeepLIFT，与博弈论的强大理论基础联系起来。*
*   *SHAP 对基于树的模型有着极快的实现，这是机器学习中最流行的方法之一。*
*   *通过计算整个数据集的 Shapely 值并聚合它们，SHAP 还可用于全局解释。它在您的本地和全球解释之间提供了一个强有力的链接。如果你使用莱姆或 SHAP 进行局部解释，然后依靠 PDP 图进行全局解释，这实际上是行不通的，你可能会得出相互矛盾的结论。*

## *实施(本地解释)*

*出于两个原因，我们在本节中将只关注 TreeSHAP:*

1.  *我们在整个博客系列中查看了结构化数据，我们选择的运行可解释性技术的模型是 RandomForest，这是一个基于树的集合。*
2.  *TreeSHAP 和 KernelSHAP 在实现中有几乎相同的接口，如果你试图解释一个 SVM，或者其他一些不是基于树的模型，用 KernelSHAP 替换 TreeSHAP 应该很简单(以计算为代价)。*

```
*import shap 
# load JS visualization code to notebook 
shap.initjs() explainer = shap.TreeExplainer(model = rf, model_output='margin') shap_values = explainer.shap_values(X_test)*
```

*这些代码行计算 Shapely 值。即使算法很快，这仍然需要一些时间。*

*   *在分类的情况下， *shap_values* 将是数组的列表，并且列表的长度将等于类的数量*
*   **explainer.expected_value* 也是如此*
*   *因此，我们应该选择我们试图解释的标签，并在进一步的绘图中使用相应的*形状值*和*预期值*。根据实例的预测，我们可以选择相应的 SHAP 值并绘制它们*
*   *在回归的情况下，shap_values 将只返回一个项目。*

*现在我们来看看个别的解释。我们将接受和石灰一样的箱子。SHAP 图书馆有多种绘制个人解释的方式——力量图和决定图。两者都非常直观地理解不同的功能一起发挥，以达到预测。如果特征的数量太大，决策图在解释时会有一点优势。*

```
*shap.force_plot( base_value=explainer.expected_value[1], shap_values=shap_values[1][row], features=X_test.iloc[row], feature_names=X_test.columns, link="identity", out_names=">50k", ) # We provide new_base_value as the cutoff probability for the classification mode 
# This is done to increase the interpretability of the plot shap.decision_plot( base_value=explainer.expected_value[1], shap_values=shap_values[1][row], features=X_test.iloc[row], feature_names=X_test.columns.tolist(), link="identity", new_base_value=0.5, )*
```

*![](img/9df50afffbf6c03b9df5425874455ae0.png)**![](img/0c95854eb05511bd1be9ade9616be516.png)*

*Force Plot*

*![](img/a95a92c4ca3dd9d95a1b60218fa15595.png)*

*Decision Plot*

*现在，我们将检查第二个示例。*

*![](img/7fa6ca897672fca5bca1e181454c566c.png)**![](img/bcec1f17b42912d33dc11d1e86993b73.png)**![](img/3e5c4951a94fde87f940b11fcd56881a.png)*

## *解释*

**例 1**

*   *与 LIME 类似，SHAP 也认为婚姻状况、年龄、教育程度等对他有很大影响。*
*   *力图解释了不同的要素如何推动和拉动输出，以将其从 *base_value* 移动到预测。这里的预测是这个人赚到> 50k 的概率或可能性(因为这是对这个实例的预测)。*
*   *在原力剧情中，你可以看到*的婚姻状况、学历数量*等。在左侧，将预测值推至接近 1，并将 hours_per_week 推至相反方向。我们可以看到，从基值 0.24 开始，这些功能通过推和拉使输出达到 0.8*
*   *在决策情节中，画面更清晰一点。你有很多特征，比如职业特征和其他特征，这使得模型输出较低，但是来自*年龄、职业 _ 执行-管理、教育 _ 数量和婚姻 _ 状态*的强烈影响已经将指针一路移动到 0.8。*
*   *这些解释符合我们对这一过程的心理模型。*

**例 2**

*   *这是一个我们分类错误的案例。这里我们解释的预测是这个人赚了 5 万。*
*   *力图显示了双方之间的均匀匹配，但是模型最终收敛到 0.6。教育程度，工作时间，以及这个人不在执行管理职位的事实都试图增加挣 5 万英镑的可能性。*
*   *如果比较两个例子的决策图，这一点就更清楚了。在第二个例子中，你可以看到一个强有力的之字形模式，在这个模式中，多个强有力的影响者推动和拉动，导致与先前信念的偏差较小。*
*   *实施(全球解释)*

## *SHAP 库还提供了简单的方法来聚合和绘制一组点(在我们的例子中是测试集)的 Shapely 值，以对模型进行全局解释。*

*解释*

```
*#Summary Plot as a bar chart 
shap.summary_plot(shap_values = shap_values[1], features = X_test, max_display=20, plot_type='bar') #Summary Plot as a dot chart 
shap.summary_plot(shap_values = shap_values[1], features = X_test, max_display=20, plot_type='dot') #Dependence Plots (analogous to PDP) 
# create a SHAP dependence plot to show the effect of a single feature across the whole dataset shap.dependence_plot("education_num", shap_values=shap_values[1], features=X_test) shap.dependence_plot("age", shap_values=shap_values[1], features=X_test)*
```

*![](img/b098e138619222671bba0a4928e2ad3d.png)*

*Summary Plot (Bar)*

*![](img/b25e78821eabd84ec694008dc1dbdfa4.png)*

*Summary Plot (Dot)*

*![](img/64c2b7f43abbceda96b0c0ece026de03.png)*

*Dependence Plot — Age*

*![](img/26caddca4ef380fd237f98617b8457fd.png)*

*Dependence Plot-Education*

## *在二元分类中，您可以绘制两个 SHAP 值中的任意一个来解释模型。我们选择了> 50k 来解释，因为这样考虑模型更直观*

*   *在汇总摘要中，我们可以看到列表顶部的常见嫌疑人。*
*   ****边注*** :我们可以给 *summary_plot* 方法提供一个 shap_values(多类分类)列表，前提是我们给*plot _ type*=‘bar’。它将以堆积条形图的形式绘制每个类别的汇总 SHAP 值。对于二进制分类，我发现这比仅仅绘制其中一个类要不直观得多。*
*   *点阵图更有趣，因为它比条形图揭示了更多的信息。除了整体重要性之外，它还显示了特征值对模型输出的影响。例如，我们可以清楚地看到，当特征值较低时，婚姻状况对积极的一面(更有可能大于 50k)产生强烈的影响。从我们的标签编码中我们知道，婚姻状态= 0，意味着已婚，1 意味着单身。所以*结婚*会增加你赚> 50k 的机会。*
*   ****边注*** :当使用 *plot_type* = 'dot 时，我们不能使用 shap 值列表。你将不得不绘制多个图表来理解你所预测的每一类*
*   *同样，如果你看一下*年龄*，你会发现当特征值较低时，它几乎总是对你赢得> 50k 的机会产生负面影响。但当特征值较高时(即你年龄较大)，会出现一个混合的点包，这告诉我们在为模型做出决策时，会有很多与其他特征的交互作用。*
*   *这就是 ***依赖图*** 出现的地方。这个想法与我们在[上一篇博文](https://deep-and-shallow.com/2019/11/16/interpretability-cracking-open-the-black-box-part-ii/)中回顾的 PD 图非常相似。但是我们用 SHAP 值来绘制相关性，而不是部分相关性。这种解释除了细微的改动之外，仍然是相似的。*
*   *X 轴表示要素的值，Y 轴表示 SHAP 值或其对模型输出的影响。正如您在点阵图中看到的，正面影响意味着它将模型输出推向预测方向(在我们的例子中大于 50k)，负面影响意味着相反的方向。所以在年龄依赖图中，我们可以看到我们之前讨论过的相同现象，但是更清楚。当你年轻时，这种影响大多是负面的，当你年老时，这种影响是正面的。*
*   *《SHAP》中的依赖情节还做了另外一件事。它选择另一个与我们正在研究的特征交互最多的特征，并根据该特征的特征值给点着色。在年龄的情况下，该方法选择的是婚姻状况。我们可以看到，你在年龄轴上发现的大部分离差是由婚姻状况解释的。*
*   *如果我们看看教育的依赖图(这是一个有序的特征)，我们可以看到教育程度越高，你挣得越多的预期趋势。*
*   *群里的小丑*

## *和往常一样，为了有效地使用这种技术，我们应该意识到它的缺点。如果你一直在寻找解释的完美技巧，很抱歉让你失望了。生活中没有什么是完美的。所以，我们来深入探讨一下缺点。*

***计算密集型**。TreeSHAP 在一定程度上解决了这个问题，但与我们讨论的大多数其他技术相比，它仍然很慢。KernelSHAP 速度很慢，对于大型数据集的计算变得不可行。(虽然有一些技术，如在计算 Shapely 值之前使用 K-means 聚类来减少数据集，但它们仍然很慢)*

*   *SHAP 价值观可能会被**曲解**，因为它不是最直观的想法。它代表的不是预测中的实际差异，而是实际预测和平均预测之间的差异，这是一个微妙的细微差别。*
*   *SHAP 和 T4 不会像莱姆一样创造稀疏的解释。人类更喜欢稀疏的解释，这些解释很好地符合心智模型。但是增加一个像 LIME 那样的正则化项并不能保证匀称的值。*
*   *KernelSHAP **忽略特性依赖**。就像排列重要性、LIME 或任何其他基于排列的方法一样，KernelSHAP 在试图解释模型时会考虑“不太可能”的样本。*
*   *额外收获:文本和图像*

# *我们讨论的一些技术也适用于文本和图像数据。虽然我们不会深入探讨，但我会链接到一些笔记本，告诉你如何做。*

*[**文本数据上的石灰—多标签分类**](https://marcotcr.github.io/lime/tutorials/Lime%20-%20multiclass.html)*

*[**图像分类上的石灰— INCEPTION_V3 — KERAS**](https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20Image%20Classification%20Keras.ipynb)*

*![](img/7da76aefd258f220ef1bb50f8fef2e1e.png)*

*[**深度讲解者——SHAP-MNIST**](https://github.com/slundberg/shap/blob/master/notebooks/deep_explainer/Front%20Page%20DeepExplainer%20MNIST%20Example.ipynb)*

*![](img/0ff40ea58f8c114f32e2e5cae39f1d47.png)*

*[**渐变讲解者—SHAP—IMAGENET 中 VGG16 的中间层**](https://github.com/slundberg/shap/blob/master/notebooks/gradient_explainer/Explain%20an%20Intermediate%20Layer%20of%20VGG16%20on%20ImageNet.ipynb)*

*![](img/ab315e62c681ab64a9f7d84d4079f2a7.png)*

*最后的话*

*![](img/c0e0431291e927a0f4a633e6f3362da4.png)*

# *我们已经到达了可解释世界旅程的终点。可解释性和可解释性是商业采用机器学习(包括深度学习)的催化剂，我们从业者有责任确保这些方面得到合理有效的解决。人类盲目信任机器还需要很长一段时间，在那之前，我们将不得不用某种可解释性来取代出色的表现，以培养信任。*

*如果这一系列的博客文章能让你回答至少一个关于你的模型的问题，我认为我的努力是成功的。*

*我的 [Github](https://github.com/manujosephv/interpretability_blog) 中有完整的代码*

***博客系列***

*[第一部](https://link.medium.com/M0YZX7zbA1)*

*   *[第二部分](/interpretability-cracking-open-the-black-box-part-ii-e3f932b03a56)*
*   *第三部分*
*   *参考*

## *Christoph Molnar，“[可解释的机器学习:使黑盒模型可解释的指南](https://christophm.github.io/interpretable-ml-book/)”*

1.  *马尔科·图利奥·里贝罗，萨米尔·辛格，卡洛斯·盖斯特林，“我为什么要相信你？:解释任何分类器的预测”，[arXiv:1602.04938](https://arxiv.org/abs/1602.04938)【cs。LG]*
2.  *《n 人游戏的价值》对博弈论的贡献 2.28(1953):307–317*
3.  *trumbelj，e .，& Kononenko，I. (2013 年)。[用特征贡献解释预测模型和个体预测](https://www.semanticscholar.org/paper/Explaining-prediction-models-and-individual-with-%C5%A0trumbelj-Kononenko/8fd17bf36bc22477bb2237c2be6e3212b753969d)。*知识和信息系统，41* ，647–665。*
4.  *斯科特·m·伦德伯格和李秀英。[“解释模型预测的统一方法。”](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf)神经信息处理系统的进展。2017.*
5.  *Lundberg，Scott M .，Gabriel G. Erion，和 Su-In Lee。"[树集合的一致个性化特征属性](https://arxiv.org/abs/1802.03888)"arXiv 预印本 arXiv:1802.03888 (2018)。*
6.  **原载于 2019 年 11 月 24 日 http://deep-and-shallow.com**的* [*。*](https://deep-and-shallow.com/2019/11/24/interpretability-cracking-open-the-black-box-part-iii/)*

**Originally published at* [*http://deep-and-shallow.com*](https://deep-and-shallow.com/2019/11/24/interpretability-cracking-open-the-black-box-part-iii/) *on November 24, 2019.**