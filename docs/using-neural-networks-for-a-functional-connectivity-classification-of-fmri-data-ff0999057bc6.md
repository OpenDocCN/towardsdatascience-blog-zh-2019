# 使用神经网络对 fMRI 数据进行功能连接分类

> 原文：<https://towardsdatascience.com/using-neural-networks-for-a-functional-connectivity-classification-of-fmri-data-ff0999057bc6?source=collection_archive---------7----------------------->

## 使用简单神经网络模型进行基于功能连接的分类的分步指南。

**在本教程中，您将:**

*   探索来自 nilearn 的神经成像数据集；
*   在 nilearn 中执行规范的独立成分分析；
*   提取由成分分析确定的区域之间的功能连接系数；
*   使用 Keras 将系数用于分类多层感知器模型。

所有必要的代码将在本指南中提供。完整版请见[本 GitHub 页面](https://github.com/gelanat/network_neuroscience/blob/master/nn_2_final.ipynb)。

***先决条件*** :本教程使用 Python、Nilearn、Keras。您不需要以前使用过 Nilearn，但是您应该有一些使用 Python 和 Keras 的经验。如果你以前从未制作过神经网络，那么当我们建立一个模型时，这个教程可能很难理解到最后，但是请不要离开 Nilearn 部分。

# 目的

我们将探索一种神经网络方法来分析关于[注意缺陷多动障碍](https://www.nhs.uk/conditions/attention-deficit-hyperactivity-disorder-adhd/)的基于功能连接的数据。功能连接显示了大脑区域如何相互连接并组成功能网络。因此，它可能会洞察大脑如何交流

我们的方法不直接依赖于神经成像扫描，而是利用矢量化的功能连接性测量。这为使用机器学习进行功能连接性分析提供了一种计算简单的方法。

使用更复杂的神经网络模型的类似方法已经被用于分类，特别是在脑障碍的情况下。然而，由于数据中的众多特征和验证难度，这些已被证明是具有挑战性的(参见 [Du 等人，2018](https://www.frontiersin.org/articles/10.3389/fnins.2018.00525/full) 的综述)。我们自己的分析也显示了不太理想的模型精度，但是本教程的目的是提供一个如何开始的指南。鼓励你最终尝试自己的神经网络！

# 数据集

Nilearn 是我们将使用的用于神经成像数据的 Python 模块，具有各种预处理数据集，您可以通过内置函数轻松下载:

```
from nilearn import datasetsnum = 40
adhd_data = datasets.fetch_adhd(n_subjects=num)
```

我们将使用 ADHD-200 公共可访问数据集，该数据集由静息态 fMRI 和从多个研究中心收集的解剖数据组成。Nilearn 只有 40 名受试者的数据，所以我们加载了所有的数据。

我们可以通过查看**来检查数据集。keys()** —我们看到有 4 种类型的信息。“func”以通向 rs-fMRI 数据图像的路径为特征；“混杂”是 CSV 文件，包含我们希望了解的不影响我们分析的讨厌变量(混杂)；“表型”为预处理步骤提供了解释；“描述”是对数据集的描述。

![](img/91e4f7cc3961bea20d73e5d6a8b88d68.png)

Examining the **.keys()** of the dataset object.

收集这些数据是为了增加对 ADHD 的神经相关性的理解。如果你愿意，你可以在这里了解更多关于原始数据集[。出于本教程的目的，您只需要知道数据集既包括典型发育个体(“对照”)，也包括诊断为 ADHD 的个体(“治疗”)。](http://fcon_1000.projects.nitrc.org/indi/adhd200/index.html)

我们将从该数据集中提取功能连接系数，以对给定受试者是“对照”还是“治疗”进行分类。我们将首先使用来自 nilearn 的独立成分分析，然后使用它来提取功能连接系数。最后，我们将使用这些系数建立一个神经网络来区分“对照组”和“治疗组”。

# 第一步:分解

独立成分分析(ICA)通常用于评估功能连接性。Nilearn 有一个用于群体水平 ICA (CanICA)的方法，该方法允许控制单个受试者的可变性，尤其是在我们对功能网络感兴趣的情况下。

我们使用 Nilearn 的内置函数，并获得我们正在处理的内容的可视化效果。我们根据 Nilearn 文档中为所选数据集提供的标准，选择使用 20 个组件的分解。我们使用 **masker_ 得到独立的组件。inverse_transform** ,然后我们使用 Nilearn 的绘图选项绘制统计和概率地图。我们需要统计方法来绘制默认模式网络(DMN)——plot _ stat _ map 函数允许绘制感兴趣区域的切割图；概率方法简单地使用从分解中产生的所有成分，并将它们层叠在默认的解剖脑图像之上。

```
from nilearn import decompositioncanica = decomposition.CanICA(n_components=20, mask_strategy=’background’)
canica.fit(func)#Retrieving the components
components = canica.components_#Using a masker to project into the 3D space
components_img = canica.masker_.inverse_transform(components)#Plotting the default mode network (DMN) without region extraction
plotting.plot_stat_map(image.index_img(components_img, 9), title='DMN')
plotting.show()#Plotting all the components
plotting.plot_prob_atlas(components_img, title='All ICA components')
plotting.show()
```

![](img/5e3490b487dcb7b5a5a02e589e5b3f9d.png)

CanICA decomposition for default mode network and all individual components.

我们的成分分解很难得到决定性的解释(本质上，我们在 fMRI 数据中看到了不同的大脑区域)，但我们可以使用它作为过滤器来提取我们感兴趣的区域。我们这样做是为了调用来自 Nilearn 的 **NiftiMapsMasker** 函数来“总结”我们使用 ICA 获得的大脑信号。一旦有了这些，我们就通过使用 **fit_transform** 方法将提取的数据转换成时间序列。

然后，我们使用我们所知道的关于数据集的一切(“func”、“混杂”和“表型”文件)来获得我们需要的信息，包括受试者是“治疗组”还是“对照组”以及它们相关的数据收集位置(地点)。

```
#Using a filter to extract the regions time series 
from nilearn import input_data
masker = input_data.NiftiMapsMasker(components_img, smoothing_fwhm=6,
 standardize=False, detrend=True,
 t_r=2.5, low_pass=0.1,
 high_pass=0.01)#Computing the regions signals and extracting the phenotypic information of interest
subjects = []
adhds = []
sites = []
labels = []
for func_file, confound_file, phenotypic in zip(
 adhd_data.func, adhd_data.confounds, adhd_data.phenotypic):
 time_series = masker.fit_transform(func_file, confounds=confound_file)
 subjects.append(time_series)
 is_adhd = phenotypic[‘adhd’]
 if is_adhd == 1:
 adhds.append(time_series) 
 sites.append(phenotypic[‘site’])
 labels.append(phenotypic[‘adhd’])
```

到目前为止，我们使用 CanICA 来获得确定感兴趣区域所需的组件。在我们能够建立我们的神经网络模型之前，最后要做的事情是获得功能连接系数。为此，我们需要查看我们提取的感兴趣区域之间的功能连接。我们考虑了三种不同的功能连接，并确定 ***相关性*** 是最准确的。你可以在[的完整代码中找到我们是如何做到的。](https://github.com/gelanat/network_neuroscience/blob/master/nn_2_final.ipynb)

相关性简单地确定成对感兴趣区域之间的边缘连通性。Nilearn 有一个计算相关矩阵的内置方法，即 **ConnectivityMeasure** 函数。我们只需要指定我们感兴趣的功能连接的种类，然后拟合我们在上一步中提取的时间序列数据。

```
from nilearn.connectome import ConnectivityMeasurecorrelation_measure = ConnectivityMeasure(kind=’correlation’)
correlation_matrices = correlation_measure.fit_transform(subjects)for i in range(40):
 plt.figure(figsize=(8,6))
 plt.imshow(correlation_matrices[i], vmax=.20, vmin=-.20, cmap=’RdBu_r’)
 plt.colorbar()
 plt.title(‘Connectivity matrix of subject {} with label {}’.format(i, labels[i]))
```

我们现在有了所有受试者的连接矩阵，但是让我们看看所有受试者的平均连接情况。我们将这些矩阵分为治疗组和对照组进行比较。这种比较是我们的神经网络模型将用于分类的。

```
#Separating the correlation matrices between treatment and control subjects
adhd_correlations = []
control_correlations = []
for i in range(40):
    if labels[i] == 1:
        adhd_correlations.append(correlation_matrices[i])
    else:
        control_correlations.append(correlation_matrices[i])#Getting the mean correlation matrix across all treatment subjects
mean_correlations_adhd = np.mean(adhd_correlations, axis=0).reshape(time_series.shape[-1],
                                                          time_series.shape[-1])#Getting the mean correlation matrix across all control subjects
mean_correlations_control = np.mean(control_correlations, axis=0).reshape(time_series.shape[-1],
                                                          time_series.shape[-1])#Visualizing the mean correlation
plotting.plot_matrix(mean_correlations_adhd, vmax=1, vmin=-1,
                               colorbar=True, title='Correlation between 20 regions for ADHD')plotting.plot_matrix(mean_correlations_control, vmax=1, vmin=-1,
                               colorbar=True, title='Correlation between 20 regions for controls')
```

![](img/b0692aaa67ac0323904bb1504b3818cf.png)![](img/216fca256bc6a483d9fac9b2caec9cf4.png)

Connectivity matrices for ADHD subjects versus controls.

我们可以看到，这两个组的连接都不是特别强(对角线可以忽略，因为它显示了与自身的相关性，因此总是等于 1)。为了更好地将联系和差异可视化，我们可以将它们投射回大脑。

```
#Getting the center coordinates from the component decomposition to use as atlas labels
coords = plotting.find_probabilistic_atlas_cut_coords(components_img)#Plotting the connectome with 80% edge strength in the connectivity
plotting.plot_connectome(mean_correlations_adhd, coords,
                         edge_threshold="80%", title='Correlation between 20 regions for ADHD')plotting.plot_connectome(mean_correlations_control, coords,
                         edge_threshold="80%", title='Correlation between 20 regions for controls')
plotting.show()
```

这给了我们一个很好的连接体，我们正在观察的 20 个区域所有连接的大脑地图。

![](img/71131a613a3507885fcbee208c31672e.png)

Connectome showing correlations between the 20 regions yielded from the component analysis. Displayed are ADHD subjects on top and control subjects on the bottom.

与对照组相比，产生的 ADHD 连接似乎不那么密集，这可能与 ADHD 相关的功能连接减少的概念有关(Yang 等人，2011)。与之前的一些研究(Tomasi & Volkow，2012)一致，对于 ADHA 受试者，我们注意到上顶叶皮层(第一个冠状图的右上部分)的连接较少，这被认为与注意力有关。DMN 中似乎也有更少的连接，我们之前想象过，这是一个在休息时活跃并与“自我”相关联的网络——一个被认为在 ADHD 中发生改变的网络(Mowinckel 等人，2017)。这些虽然很小的差异，但表明在我们的神经网络模型中使用相关矩阵对治疗组和对照组进行分类应该不是不可能的。

如果您希望看到连接体的交互式可视化，您可以运行下面的代码行。否则，我们就可以开始建模了！

```
#Creating the interactive visualization
view = plotting.view_connectome(mean_correlations, coords, edge_threshold='80%')#To display in the cell below
view#To display in a different tab
view.open_in_browser()
```

# 神经网络方式

既然我们的相关矩阵提供了功能连接的矢量化测量，我们可以使用这些作为神经网络的输入数据。

在我们构建模型之前，我们应该将数据分为训练数据(70%)和测试数据(30%):

```
from sklearn.model_selection import train_test_splitX_train, X_test, y_train, y_test = train_test_split(correlation_matrices, labels, test_size=0.3)
```

我们的神经网络可以是从一层到五层的任何东西。在尝试了几种不同的架构之后，我们选定了由四个**密集**层组成的**顺序**模型。本教程假设您知道这些意味着什么，所以我们不会深入所有的细节，而是给出一个整体架构的简要概述。如果你需要重温这个话题，[这个博客](/building-a-deep-learning-model-using-keras-1548ca149d37)是一个好的开始。

```
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adamclassifier = Sequential()#First Hidden Layer
classifier.add(Dense(32, activation=’tanh’, kernel_initializer=’random_normal’, input_shape=connectivity_biomarkers[‘correlation’].shape[1:]))#Second Hidden Layer
classifier.add(Dense(16, activation=’relu’, kernel_initializer=’random_normal’))#Third Hidden Layer
classifier.add(Dense(16, activation=’relu’, kernel_initializer=’random_normal’))#Output Layer
classifier.add(Dense(1, activation=’sigmoid’, kernel_initializer=’random_normal’))#Compiling the model
classifier.compile(optimizer = Adam(lr =.0001),loss='binary_crossentropy', metrics =['accuracy'])#Fitting the model
classifier.fit(np.array(X_train),np.array(y_train), batch_size=32, epochs=100)
```

这就是了！我们建立了一个简单的神经网络。让我们回顾一下我们模型的架构:

*   我们使用**顺序**模型，这样我们可以简单地在一层之上构建另一层；
*   我们选择**密集**层，它们是神经网络中的简单层——你可以把它们想象成接受多个输入并产生单个输出的线性模型。当我们处理不一定是线性可分的函数时，我们使用其中的 4 个，但是，由于我们本质上只对二元分类感兴趣，所以您不必使用 4 个，并且可能用更少的就可以了——就模型的预测能力而言，这似乎是最有效的。对于每一层，我们指定不同数量的节点(32、16、16、1 ),这在这个示例模型中也是微不足道的。唯一的经验法则是从小于或等于输入长度的数字开始；在我们的例子中，我们有 40 个矩阵，所以 32 个似乎是合适的。最终的输出图层应该与输出类别的数量相关。由于这是一个二元分类问题，我们需要使用 1。
*   对于激活功能(主要完成输入输出信号转换)，我们使用 **Tanh** 、 **ReLu** 和 **Sigmoid** 。 **Tanh** 和 **Sigmoid** 分别用于第一层和最后一层，不用于隐藏层。这是因为两者都可以通过消失梯度问题来表征，这意味着如果输入更高，它们将输出零梯度(我们不确定，所以最好是安全的)。我们本可以使用 **ReLu** 函数来克服这个问题，因为它很简单(如果 **x** 为正，它给出 **x** 作为输出，否则为 0)，但在非隐藏层中最好避免使用这些函数，因为它们可能不会产生梯度或“死亡”神经元。当前的架构( **Tanh，ReLu，ReLu，Sigmoid** )似乎又一次产生了最佳的评估指标，比如准确性；
*   最后，我们使用一个 **Adam** 优化器和一个**二元交叉熵**损失函数。 **Adam** 被认为是一个不错的选择，尤其是对于有噪声的数据。**二进制交叉熵**是分类问题的常用选择，因为它独立于每个类别，并确保一个输出向量不受其他分量的影响。

![](img/926626be96fc1966b278d782c2409156.png)

Example four Dense layer Sequential neural network model architecture.

现在，只剩下一件事要做了——看看我们的模型表现如何。我们将使用准确性作为我们的主要衡量标准，因为它代表了正确分类的比例。

让我们从训练集开始:

```
eval_model=classifier.evaluate(np.array(X_train), np.array(y_train))
eval_model
```

万岁，我们训练数据的准确率是 1。测试数据的可怕部分。

```
y_pred=classifier.predict(X_test,batch_size=32)
y_pred =(y_pred>0.5)from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
print(cm)
cr = classification_report(y_test, y_pred)
print(cr)
```

![](img/3c31060f1a3940782f99237f54b84425.png)

Confusion matrix and classification report for our model.

我们分类的总体准确率是 75%,这并不可怕，但还可以更好。我们也只得到 2 个假阴性和 1 个假阳性。现在就看你如何使用这个分步指南中构建的简单模型了！免责声明:相应的 [Jupyter 笔记本](https://github.com/gelanat/network_neuroscience/blob/master/functional%20connectivity%20using%20neural%20nets.ipynb)中的模型比这里报告的精度低得多，可能是由于随机过程——这就更有理由进行实验并提出更好的模型。

本教程探讨了如何使用功能连接数据来筛查多动症。为此，我们根据不同区域的功能连接系数创建了一个神经网络。正如我们通过成分分析和连接体图所看到的，治疗和对照样品之间的差异并不显著，这可以解释我们筛选的准确性较低。应该对更大的数据集做更多的工作，以观察 ADHD 的功能连接是否有更一致的模式。在未来的分析中研究更多的区域和组成部分也是有意义的。

希望这已经给了你一些如何使用机器学习进行功能性磁共振成像数据的功能连接分析的想法。我们(方便地)使用了预处理数据，但如果你想了解更多关于 fMRI 数据分析的预处理步骤，请查看我的另一篇教程[这里](https://medium.com/@gelana/using-fmriprep-for-fmri-data-preprocessing-90ce4a9b85bd)。下次见！

# 参考

Abraham，a .，Pedregosa，f .，Eickenberg，m .，Gervais，p .，Mueller，a .，Kossaifi，j .，… & Varoquaux，G. (2014 年)。用 scikit-learn 进行神经成像的机器学习。*神经信息学前沿*、 *8* 、14。

杜，杨，傅，钟，和卡尔豪，V. D. (2018)。使用功能连接对脑疾病进行分类和预测:有希望但有挑战。*神经科学前沿*、 *12* 。

Mowinckel，A. M .，Alnæ，d .，Pedersen，M. L .，Ziegler，s .，Fredriksen，m .，Kaufmann，t .，… & Biele，G. (2017)。默认模式可变性的增加与任务执行能力的降低有关，这在患有 ADHD 的成年人中很明显。*神经影像:临床*， *16* ，369–382。

Pedregosa，f .，Varoquaux，g .，Gramfort，a .，Michel，v .，Thirion，b .，Grisel，o .，… & Vanderplas，J. (2011 年)。sci kit-learn:Python 中的机器学习。*机器学习研究杂志*，*12*(10 月)，2825–2830。

托马斯博士和沃尔考博士(2012 年)。注意缺陷/多动障碍儿童的异常功能连接。*生物精神病学*， *71* (5)，443–450。

瓦洛夸、萨达吉阿尼、皮内尔、克莱因施密特、波林、J. B .、和蒂里翁(2010 年)。fMRI 数据集上稳定的多主题 ICA 的分组模型。*神经影像*， *51* (1)，288–299。

杨海红，吴，秦群，郭立涛，李，秦群，龙，黄小群，… &龚清友(2011)。首次接受药物治疗的 ADHD 儿童的异常自发脑活动:静息态 fMRI 研究。*神经科学快报*， *502* (2)，89–93。