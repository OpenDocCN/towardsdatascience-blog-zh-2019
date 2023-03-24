# 梯度观点中的正则化[张量流中的手动反向传播]

> 原文：<https://towardsdatascience.com/regularization-in-gradient-point-of-view-manual-back-propagation-in-tensorflow-b7811ad297da?source=collection_archive---------21----------------------->

## 正则化对渐变有什么影响？

![](img/a7edd065f6295796bc702d45b8273553.png)

GIF from this [website](https://giphy.com/gifs/cat-money-cash-ND6xkVPaj8tHO)

正则化就像上面展示的那只猫，当一些权重想要变得“大”时，我们惩罚它们。今天我想看看正则项给梯度带来了什么样的变化。下面是我们将要比较的不同正则化术语的列表。(θ是每层的重量。).

***Z .基线*** *(无正则化)* ***a .θ****【岭回归】* ***b . ABS(θ)+sqrt(ABS(θ)****(*[*弹性网*](https://web.stanford.edu/~hastie/Papers/B67.2%20%282005%29%20301-320%20Zou%20&%20Hastie.pdf)*)* ***c .θ*** *(来自论文“* [*【比较稀疏度】*](https://arxiv.org/abs/0811.4706)*”* ***e . ABS(θ)****(p 值为 1 的 p-norm)* ***f . sqrt(ABS(θ))****(p-norm 带 p-的* *(来自论文“*[](https://arxiv.org/abs/0811.4706)**”* ***I .-tanh(θ)****(来自论文“* [*比较稀疏度*](https://arxiv.org/abs/0811.4706)*”* ***j****

***简介***

*为了防止过拟合，或者控制过拟合的程度(过拟合的严重程度),我们使用正则化，例如去掉或增加 L1 或 L2 正则化项。正则化背后的一般思想(除了退出)，是在最小化目标函数的同时最小化权重的大小，当然，一些正则化旨在创建‘稀疏’权重，这意味着大多数权重值为零。*

*接下来，对于神经网络架构中的任何训练过程，都有一些共同的元素，例如激活前后的值。相对于权重的梯度，以及传递到前面层的梯度。权重本身，以及使用 Adam 优化器时每个权重的动量值。*

*所以把两个和两个放在一起，正则项对那些值有什么样的影响？它们的均值、标准差、偏度、峰度和非零元素的数量，它们会随着基线表现增加、减少还是保持不变？*

***相关作品***

*关于如何解释正则化项，最有趣的观点之一是将其视为权重之前的贝叶斯。这篇[的博文](http://rohanvarma.me/Regularization/)详细描述了我们理解正则化术语的不同观点。(我怀疑正则化的程度取决于先验的知情/不知情程度，我还认为随着训练的进行，最终可能性将对权重产生更大的影响。).*

*此外，如果有人对 L1 和 L2 正规化背后的基本直觉以及一些命名历史感兴趣，请查看此[博客帖子](https://blog.alexlenail.me/what-is-the-difference-between-ridge-regression-the-lasso-and-elasticnet-ec19c71c9028)。*

***实验设置***

*![](img/428f2b59f788629b52a21ea640499874.png)*

***蓝色球体** →来自 [STL 数据集的输入图像](https://cs.stanford.edu/~acoates/stl10/)(20，96，96，3)的形状
**黄色矩形** →带有 ReLU 激活的卷积层
**红色正方形** →交叉熵损失项*

*整个实验只是一个分类任务，根据我们添加的正则项，黄色矩形在计算梯度时会有一个附加项。在继续之前，我希望澄清我将要用到的每个单词的意思。*

*![](img/ee8efe5cf97b9355e8f84ae0582cd56f.png)**![](img/bc4b9e44d32a5516b4f3481e62f284fd.png)*

***结果***

> *训练/测试准确性*

*![](img/132a22041c1b41519e6b67d9a4c5f374.png)**![](img/cc433654ed75a07a8c61f9a947c60e5d.png)*

*当我们查看训练和测试数据的准确度图时，我们可以观察到…..
**添加时达到最高训练精度**:***sqrt(θ)/θ***
**添加时达到最高测试精度**:***-tanh(θ)***
**添加时达到最低性能**:***θ****

*当我们添加术语 ***θ*** 时，每个权重的导数值就变成 1，我怀疑这可能会导致超调。*

> *gradientp 的平均统计值(传递到上一层的梯度)*

*![](img/c97c32b5851fd1ce179881a0c46a49c9.png)*

*当我们查看传递到上一层的渐变值时，有一种颜色非常突出，那就是灰色。加上 ***-log(1+θ )*** 这一项，似乎让梯度值变化很大。*

> *梯度的平均统计值 w(相对于重量的梯度)*

*![](img/904b30f88254abfc56016bdd954fa901.png)*

*当我们将注意力集中在非零元素的数量上时，我们观察到，在没有任何正则化项的情况下，关于每个权重的梯度主要由零组成。(这也可能是造成收敛慢的原因。).*

*当我们把注意力集中在平均标准偏差值上时，我们注意到在训练的最后区域，粉红色和黄色的线相交。然而，案例 H(粉色)在训练/测试数据上的表现比案例 B(黄色)好得多。*

> *层的平均统计数据(激活前的层值)*

*![](img/2073e1a3f3921104d159a5d076e5c654.png)*

*当我们查看每一层的值(激活函数之前)时，我们马上注意到，随着训练的继续，情况 C(绿色)的平均值不断下降。我不认为这种减少本身是一件坏事，但是，与其他情况相比，这是极端的，另外请注意，我们正在使用 ReLU 激活，这意味着任何小于零的值，都将被丢弃并被零取代。*

> *层 a 的平均统计数据(激活后的层值)*

*![](img/497c0a2941b3b7b7f08265ddceccc805.png)*

*我们已经知道，添加 ***-tanh(θ)*** (粉红色线)在测试图像上给了我们最好的结果，当我们将注意力集中在激活层之后的非零元素的数量时，情况 H(粉红色线)在其自己的区域中。*

*当其他案例高于或低于 60，000 时，奇怪的是案例 H 似乎集中在该区域。与其他情况相比，数据的内部表示更加稀疏。*

> *重量的平均统计(每种情况下的重量值)*

*![](img/f7fb84fde86c0f479453f2becc5391a0.png)*

*由于情况 C(绿线)过冲，我们得到了一个扭曲的图表，其中其他值被压缩，以适应情况 C 中重量的减少，因此在下一个图中，我已将情况 C 从平均重量图中移除。*

*![](img/0e1a1f0f8e4ea11df5aebd670e07ca32.png)*

*移除后，我们可以清楚地看到不同权重值如何随时间变化。同样，我们注意到情况 H(粉线)位于它自己的区域，我不知道它为什么这样做，也没有意义，因为正则化的要点是最小化权重，为什么值比其他情况高？*

> *矩的平均统计数据(Adam 优化器的动量部分)*

*![](img/26b6e999addd7bbb827b7a54834465be.png)*

*从上面的图中，我们(也许能够)得出结论，情况 C(绿线)已经过冲。好像都是？或者说动量的大部分值为零。标准差都是零，表明动量内没有太多的变化。*

***讨论***

*我想以这句话开始这一部分，*

> ***我不能得出任何结论，这只是**一个**实验，因此从一个样本得出任何结论都是错误的。此外，我的实验的一个关键缺陷是，所有的权重没有被同等地初始化！我给出了完全相同的种子值和标准偏差(如下所示),但我不确定是否所有的权重都被同样初始化了。***

*![](img/e65c73aac7d103cbdd8750a4d94fb960.png)*

*从所有这些实验来看，似乎没有规则可以保证一个模型在测试数据上表现良好。(或者会有更高的泛化能力)。然而，似乎有一个模型可以很好地概括的独特模式。*

*我将只关注情况 H(粉红色)，通常，我们理解正则化是为了限制权重大小的增长，然而，从上图中我们可以清楚地看到情况 H 权重的平均值高于其他情况…所以…？问题就变成了它规范得好吗？使用 tanh 术语可能是这背后的原因吗？*

*但是另外，我们注意到在 ReLU 激活后，案例 H 的 layera 值比其他案例更稀疏。这意味着健康剂量的零被混合在内部表示中。我真的很想知道为什么案例 H 给出了最好的测试精度，目前只能用“因为它被很好地正则化了”来解释，但是我知道还有更多。*

***结论/代码***

*![](img/2d71675bb4fadb062c6a12378187217c.png)*

*总之，权重大小和稀疏性之间的正确平衡似乎给出了良好的泛化性能。(手动波浪形)。*

*要访问 google collab 中的代码，请[点击此处](https://colab.research.google.com/drive/1C13_mV9zChz6eQwZgZrPZHLvQ_b2I6C7)，要访问我的 GitHub 中的代码，请[点击此处](https://github.com/JaeDukSeo/Daily-Neural-Network-Practice-2/blob/master/Class%20Stuff/Sparsity%20in%20Gradient%20Point%20of%20View/blog/a%20blog.ipynb)。*

***遗言***

*在做这些实验时，我不禁想到这些类型的正则化有一个(不那么关键的)缺陷。*

> *它们都是被动正规化方法。*

*更具体地说，网络中的每个权重，不管它们位于何处，它们倾向于捕捉什么样的特征，它们都被正则化。我当然相信这是一个有趣的研究领域，例如，我们可以有一个简单的条件语句，即“如果权重的大小大于 x，则执行正则化”或更复杂的语句“如果该权重和类标签之间的互信息已经很高，则执行正则化”。我想知道这些方法将如何改变训练的整体动力。*

*最后，我要感谢我的导师，Bruce 博士给我推荐了论文“[稀疏度的比较](https://arxiv.org/abs/0811.4706)”。更多文章请访问我的[网站](https://jaedukseo.me/)。*

***附录(每种情况下每层的 GIF 动画)***

> *情况 Z:按梯度、梯度、层、层 a、力矩、重量的顺序*

*![](img/8e29bcd0dc6206a700deee96c17422c9.png)**![](img/7a1ae1dc28187f4c567821c4ca9a3698.png)**![](img/6245b7242b2c0bd6fef2d70bf3b5c726.png)**![](img/c4ea43c2eb68e97c7d3a06822ce011f0.png)**![](img/f4b3e680a61192bb75dce8e49e20caea.png)**![](img/6cf5804574da58bfd67b52dbe03421fd.png)*

> *情况 A:按梯度 p、梯度 w、层、层 A、力矩、重量顺序*

*![](img/ea74cb8e360b4877e150e64dce2efbde.png)**![](img/cb3e74aef90fbcf6f61c42aaa7306c30.png)**![](img/c3e13717ee71272ac80505732c8cde8a.png)**![](img/ec9fa4c62f6bef46532912211af926f3.png)**![](img/50c036574cb0df086a3ac196eeb79088.png)**![](img/3231d58d37197bdad738f07512992176.png)*

> *情况 B:按梯度 p、梯度 w、层、层 a、力矩、重量顺序*

*![](img/d8e0a6ed58c946f901f8ab5d3bf9510c.png)**![](img/1fcc52ce782160010b86eec08d990a47.png)**![](img/f136f5ac852cab8214c0425339a323db.png)**![](img/31a7e9bacc0bd09b71cee5100c56abc1.png)**![](img/c487ddf88ae4492e88d7c58267e6bd9f.png)**![](img/9787e9943cb1300d06de7982b1d73801.png)*

> *情况 C:按梯度 p、梯度 w、层、层 a、力矩、重量顺序*

*![](img/03fded24095f09c71feda4d422a02b97.png)**![](img/407d09e1d6d66f826c9e93ad6e5be7f7.png)**![](img/11fa0b1cdbf867edeee2ccd95c449f09.png)**![](img/1cc254fdc94b0bf2016e0d0d3ae398ff.png)**![](img/cf2572cc7387321be36ef95ca55c8531.png)**![](img/a780078f2c56642c8ae8ec39e7829ed2.png)*

> *情况 D:按梯度 p、梯度 w、层、层 a、力矩、重量顺序*

*![](img/ef5f45c149554afc4ef429babe532c41.png)**![](img/db997bd08a4a683a4145175c9c3cba5e.png)**![](img/f7d5cef424172fd436441a962d402f6c.png)**![](img/df34cd5250c9fd82b443985401832710.png)**![](img/309a3d510f5feb0ed169be0efa214cc9.png)**![](img/ed9f9689b07d2bb65a0f70463152e0dc.png)*

> *情况 E:按梯度 p、梯度 w、层、层 a、力矩、重量顺序*

*![](img/1dd5f124ef7a12bf4fccace1991678ca.png)**![](img/688ec49c1a4d9a53db40977eabe5d3f8.png)**![](img/6e157bb3ea7923f628b711c1d79d1a21.png)**![](img/60a1d1f49a82ac7003dfb5ca00669e71.png)**![](img/2d5defe8df7ad814bf66ad8b557d3337.png)**![](img/304981bd2c7ca84d0270ce407aba73b2.png)*

> *情况 F:按梯度 p、梯度 w、层、层 a、力矩、重量顺序*

*![](img/c163140b83e2da49d28b238c5c812e44.png)**![](img/f5c4c7b10a0da0d890b4a7977fd5aea9.png)**![](img/df1195786eab79a39d0d4b85c61b61cf.png)**![](img/aa306492a6bf064333dec55246f97dd1.png)**![](img/54761f1c4ee5a0307a86b878a8533eba.png)**![](img/c7a439966d9088ef45c34ff83358a8f4.png)*

> *情况 G:按梯度 p、梯度 w、层、层 a、力矩、重量顺序*

*![](img/6245830c8af627408b09fe80fb30f550.png)**![](img/334e5f2ac1c753bb28257f84b4ca0fa3.png)**![](img/4ae986bb7c25dfd49fd2e6b5b29ef68a.png)**![](img/6fee7357c7759a85fda022f77c0461af.png)**![](img/9f1c290b37aac539160942ac38dbee43.png)**![](img/4ffa80b17659cab2cc9462433ba652ba.png)*

> *情况 H:按梯度 p、梯度 w、层、层 a、力矩、重量顺序*

*![](img/3d382912370561a0396bfb6be95507f6.png)**![](img/8f7ab079581fa3e3114e2e56ecfe21e4.png)**![](img/1956c504da3fff7e44d1aa73d4d42681.png)**![](img/8a751d1b66fe0e7bdb2b53915f8f249c.png)**![](img/6a2494ea6ca21791bd4289414e3b4cde.png)**![](img/618c3af87135deee6afd4cba40ac2692.png)*

> *情况一:按梯度、梯度、层、层、力矩、重量的顺序*

*![](img/a5b8056e0c3e4f445eb0cabc00cdb41c.png)**![](img/0bf0ec6c40e4d1e49dc403d94227f156.png)**![](img/cff8916170f0da9bd2b87eaacec9bfe3.png)**![](img/1a21c068dc1f28f9ff5af74eabe493db.png)**![](img/6184ee0ca8045f4f1f2256fa3724c3a1.png)**![](img/b52a42f4425c13deb7d7811a7313bd7f.png)*

> *情况 J:按梯度 p、梯度 w、层、层 a、力矩、重量的顺序*

*![](img/cf5ba3b3625ac430fee1dace816cdaff.png)**![](img/1fc35097914492a889704bc048d77916.png)**![](img/4fb04bc33d8793b603fb576023e3fc90.png)**![](img/942528f6a146e1e4083fbf27e6179092.png)**![](img/b3559a655977118af8f07d8e1af529fc.png)**![](img/f6dd8be86fb427dd417b98dbc2564abf.png)*

***附录(衍生品)***

*![](img/1ad740e3fc0f1e086462780e059419f2.png)*

***参考***

1.  *n .布鲁斯(2016)。尼尔·布鲁斯。尼尔·布鲁斯。检索于 2019 年 1 月 4 日，来自[http://www.scs.ryerson.ca/~bruce/](http://www.scs.ryerson.ca/~bruce/)*
2.  *Hurley 和 s . Rickard(2008 年)。稀疏度的比较。arXiv.org。检索于 2019 年 1 月 4 日，来自[https://arxiv.org/abs/0811.4706](https://arxiv.org/abs/0811.4706)*
3.  *用 Python 进行数值和科学计算:用 Python 和 Matplotlib 创建子图。(2019).python-course . eu . 2019 年 1 月 4 日检索，来自[https://www . python-course . eu/matplotlib _ multiple _ figures . PHP](https://www.python-course.eu/matplotlib_multiple_figures.php)*
4.  *SciPy . stats . Kurt osis—SciPy v 1 . 2 . 0 参考指南。(2019).Docs.scipy.org。检索于 2019 年 1 月 4 日，来自[https://docs . scipy . org/doc/scipy/reference/generated/scipy . stats . Kurt osis . html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html)*
5.  *scipy.stats.skew — SciPy v0.13.0 参考指南。(2019).Docs.scipy.org。检索于 2019 年 1 月 4 日，来自[https://docs . scipy . org/doc/scipy-0 . 13 . 0/reference/generated/scipy . stats . skew . html](https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.stats.skew.html)*
6.  *numpy.count _ 非零—NumPy 1.15 版手册。(2019).Docs.scipy.org。2019 年 1 月 4 日检索，来自[https://docs . scipy . org/doc/numpy-1 . 15 . 1/reference/generated/numpy . count _ 非零值. html](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.count_nonzero.html)*
7.  *数组？，E. (2017)。高效计算 numpy 数组中的零元素？。堆栈溢出。检索于 2019 年 1 月 4 日，来自[https://stack overflow . com/questions/42916330/efficiency-count-zero-elements-in-numpy-array](https://stackoverflow.com/questions/42916330/efficiently-count-zero-elements-in-numpy-array)*
8.  *解析，S. (2013)。语法错误:分析时出现意外的 EOF。堆栈溢出。检索于 2019 年 1 月 4 日，来自[https://stack overflow . com/questions/16327405/syntax error-unexpected-eof-while-parsing](https://stackoverflow.com/questions/16327405/syntaxerror-unexpected-eof-while-parsing)*
9.  *Matplotlib . py plot . legend-Matplotlib 3 . 0 . 2 文档。(2019).Matplotlib.org。检索于 2019 年 1 月 4 日，来自[https://matplotlib . org/API/_ as _ gen/matplotlib . py plot . legend . html](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html)*
10.  *行与列——谷歌搜索。(2019).Google.com。2019 年 1 月 4 日检索，来自[https://www.google.com/search?q=row+vs+column&rlz = 1 C1 chbf _ enca 771 ca 771&OQ = row+vs+col&aqs = chrome . 0.35 i39 j 69 I 60j 69 I 57j 0l 3.2289j 1j 7&sourceid = chrome&ie = UTF-8](https://www.google.com/search?q=row+vs+column&rlz=1C1CHBF_enCA771CA771&oq=row+vs+col&aqs=chrome.0.35i39j69i60j69i57j0l3.2289j1j7&sourceid=chrome&ie=UTF-8)*
11.  *图例指南— Matplotlib 2.0.2 文档。(2019).Matplotlib.org。检索于 2019 年 1 月 4 日，来自[https://matplotlib.org/users/legend_guide.html](https://matplotlib.org/users/legend_guide.html)*
12.  *正则项导数的随机注释。(2019).中等。检索于 2019 年 1 月 4 日，来自[https://medium . com/@ SeoJaeDuk/archived-post-random-notes-for-derivative-for-regulation-terms-1859 B1 faada](https://medium.com/@SeoJaeDuk/archived-post-random-notes-for-derivative-for-regularization-terms-1859b1faada)*
13.  *IPython r .(2013 年)。IPython 中释放大数组内存。堆栈溢出。检索于 2019 年 1 月 5 日，来自[https://stack overflow . com/questions/16261240/releasing-memory-of-huge-numpy-array-in-ipython](https://stackoverflow.com/questions/16261240/releasing-memory-of-huge-numpy-array-in-ipython)*
14.  *内置神奇命令— IPython 7.2.0 文档。(2019).ipython . readthe docs . io . 2019 年 1 月 5 日检索，来自[https://ipython . readthe docs . io/en/stable/interactive/magics . htm](https://ipython.readthedocs.io/en/stable/interactive/magics.html)l*
15.  *j . Chapman(2017 年)。如何用 Python 制作 gif？多余的六分仪。检索于 2019 年 1 月 5 日，来自[http://superfluoussextant.com/making-gifs-with-python.html](http://superfluoussextant.com/making-gifs-with-python.html)*
16.  *岭回归、套索和弹性网之间的区别是什么？。(2017).兼收并蓄的秘传。检索于 2019 年 1 月 5 日，来自[https://blog . alexlenail . me/what ' s-the-difference-the-ridge-regression-the-lasso-and-elastic net-EC 19 c 71 c 9028](https://blog.alexlenail.me/what-is-the-difference-between-ridge-regression-the-lasso-and-elasticnet-ec19c71c9028)*
17.  *(2019).Web.stanford.edu。检索于 2019 年 1 月 5 日，来自[https://web . Stanford . edu/~ hastie/Papers/b 67.2% 20% 282005% 29% 20301-320% 20 Zou % 20&% 20 hastie . pdf](https://web.stanford.edu/~hastie/Papers/B67.2%20%282005%29%20301-320%20Zou%20&%20Hastie.pdf)*
18.  *面向数据科学的 L1 和 L2 正则化方法。(2017).走向数据科学。检索于 2019 年 1 月 5 日，来自[https://towards data science . com/L1-and-L2-regulation-methods-ce 25 e 7 fc 831 c](/l1-and-l2-regularization-methods-ce25e7fc831c)*
19.  *岭回归、套索和弹性网之间的区别是什么？。(2017).兼收并蓄的秘传。检索于 2019 年 1 月 5 日，来自[https://blog . alexlenail . me/what ' s-the-difference-the-ridge-regression-the-lasso-and-elastic net-EC 19 c 71 c 9028](https://blog.alexlenail.me/what-is-the-difference-between-ridge-regression-the-lasso-and-elasticnet-ec19c71c9028)*
20.  *STL-10 数据集。(2019).Cs.stanford.edu。检索于 2019 年 1 月 5 日，来自[https://cs.stanford.edu/~acoates/stl10/](https://cs.stanford.edu/~acoates/stl10/)*
21.  *plot，H. (2010 年)。如何更改 matplotlib 绘图的字体大小？堆栈溢出。检索于 2019 年 1 月 5 日，来自[https://stack overflow . com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot](https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot)*
22.  *matplotlib？，H. (2015)。如何使用 matplotlib 在同一行中绘制多个图形？。堆栈溢出。检索于 2019 年 1 月 5 日，来自[https://stack overflow . com/questions/34291260/how-can-I-plot-multiple-figure-in-the-same-line-with-matplotlib](https://stackoverflow.com/questions/34291260/how-can-i-plot-multiple-figure-in-the-same-line-with-matplotlib)*
23.  *正文，I. (2016)。IPython 笔记本键盘快捷键搜索文本。堆栈溢出。检索于 2019 年 1 月 5 日，来自[https://stack overflow . com/questions/35119831/ipython-notebook-keyboard-shortcut-search-for-text](https://stackoverflow.com/questions/35119831/ipython-notebook-keyboard-shortcut-search-for-text)*
24.  *创建和导出视频剪辑— MoviePy 0.2.3.2 文档。(2019).zulko . github . io . 2019 年 1 月 5 日检索，来自[https://zulko . github . io/movie py/getting _ started/video clips . html](https://zulko.github.io/moviepy/getting_started/videoclips.html)*
25.  *用 Python 制作视频文件的 gif—_ _ del _ _(self)。(2014).zulko . github . io . 2019 年 1 月 5 日检索，来自[http://zulko . github . io/blog/2014/01/23/making-animated-gifs-from-video-files-with-python](http://zulko.github.io/blog/2014/01/23/making-animated-gifs-from-video-files-with-python/)/*