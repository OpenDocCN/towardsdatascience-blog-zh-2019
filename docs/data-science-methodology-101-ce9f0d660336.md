# 数据科学方法论 101

> 原文：<https://towardsdatascience.com/data-science-methodology-101-ce9f0d660336?source=collection_archive---------4----------------------->

## 一个数据科学家如何组织他的工作？

每个数据科学家都需要一种方法来解决数据科学的问题。例如，假设你是一名数据科学家，你的第一份工作是增加一家公司的销售额，他们想知道他们应该在什么时间段销售什么产品。你需要正确的方法来组织你的工作，分析不同类型的数据，并解决他们的问题。你的客户不关心你如何工作；他们只关心你是否能及时完成。

## 什么是数据科学中的方法论？

数据科学中的方法论是组织你的工作的最佳方式，可以做得更好，并且不浪费时间。数据科学方法由 10 个部分组成:

![](img/6abf2ad50709ab9660259f32d9daf5a6.png)

Source: [https://www.ibmbigdatahub.com/blog/why-we-need-methodology-data-science](https://www.ibmbigdatahub.com/blog/why-we-need-methodology-data-science)

在本文中，有五个部分，每个部分包含更多的步骤:

1.  从问题到方法
2.  从需求到收集
3.  从理解到准备
4.  从建模到评估
5.  从部署到反馈

如果我们看最后一张图中的图表，我们会看到它是高度迭代的，永远不会结束；这是因为在真实的案例研究中，我们必须重复一些步骤来改进模型。

## 从问题到方法

每个客户的请求都始于一个问题，数据科学家的工作首先是理解它，并用统计和机器学习技术来解决这个问题。

*   **业务** **了解**阶段至关重要，因为它有助于明确客户的目标。在这个阶段，我们必须就问题的每一个方面向客户提出许多问题；通过这种方式，我们确信我们将研究相关的数据，并且在这一阶段结束时，我们将有一个**业务需求**的列表。
*   下一步是**分析方法**，在这里，一旦清楚地陈述了业务问题，数据科学家就可以定义解决问题的分析方法。这一步需要在统计和机器学习技术的背景下表达问题，这是必不可少的，因为它有助于确定需要什么类型的模式来最有效地解决问题。如果问题是确定某事的概率，那么可以使用预测模型；如果问题是显示关系，可能需要一个描述性的方法，如果我们的问题需要计数，那么统计分析是解决它的最好方法。对于每种方法，我们可以使用不同的算法。

![](img/6d38f074d9d91dad22defea05469ec53.png)

Source: [https://www.displayr.com/what-is-a-decision-tree/](https://www.displayr.com/what-is-a-decision-tree/)

## 从需求到收集

一旦我们找到了解决问题的方法，我们将需要为我们的模型发现正确的数据。

*   **数据要求**是我们为初始数据收集确定必要的数据内容、格式和来源的阶段，我们在所选方法的算法中使用这些数据。
*   在**数据收集**阶段，数据科学家确定与问题领域相关的可用数据资源。为了检索数据，我们可以在相关网站上进行网络搜集，或者使用带有现成数据集的存储库。通常，预制数据集是 CSV 文件或 Excel 无论如何，如果我们想从任何网站或存储库中收集数据，我们应该使用 Pandas，这是一个下载、转换和修改数据集的有用工具。这里有一个熊猫数据收集阶段的例子。

```
import pandas as pd # download library to read data into dataframepd.set_option('display.max_column', None)
dataframe = pd.read_csv("csv_file_url")
print("Data read into dataframe!")dataframe.head() # show the first few rows
dataframe.shape # get the dimensions of the dataframe
```

## 从理解到准备

现在数据收集阶段已经完成，数据科学家使用描述性统计和可视化技术来更好地理解数据。数据科学家探索数据集以了解其内容，确定是否需要额外的数据来填补任何空白，以及验证数据的质量。

*   在**数据理解**阶段，数据科学家试图更多地理解之前收集的数据。我们必须检查每个数据的类型，并了解更多关于属性及其名称的信息。

```
# get all columns from a dataframe and put them into a list
attributes = list(dataframe.columns.values)# then we check if a column exist and what is its name.
print([match.group(0) for attributes in attributes for match in [(re.compile(".*(column_name_keyword).*")).search(attributes)] if match])
```

*   在**数据准备**阶段，数据科学家为建模准备数据，这是最关键的步骤之一，因为模型必须清晰无误。在这一阶段，我们必须确保数据的格式对于我们在分析方法阶段选择的机器学习算法是正确的。数据帧必须有适当的列名、统一的布尔值(是、否或 1，0)。我们必须注意每个数据的名称，因为有时它们可能用不同的字符书写，但它们是同一件事；例如(water，WaTeR)，我们可以将一列中的所有值都变成小写。另一个改进是从数据帧中删除数据异常，因为它们是不相关的。

```
# replacing all 'yes' values with '1' and 'no' with '0'
dataframe = dataframe.replace(to_replace="Yes", value=1)
dataframe = dataframe.replace(to_replace="No", value=0)# making all the value of a column lowercase
dataframe["column"] = dataframe["column"].str.lower()
```

## 从建模到评估

一旦为选择的机器学习算法准备好数据，我们就准备好建模了。

*   在**建模**阶段，数据科学家有机会了解他的工作是否准备就绪，或者是否需要评审。建模侧重于开发描述性或预测性的模型，这些模型基于统计或通过机器学习采取的分析方法。**描述性建模**是一个数学过程，描述现实世界的事件以及造成这些事件的因素之间的关系，例如，描述性模型可能会检查这样的事情:如果一个人这样做，那么他们可能会更喜欢那样。**预测建模**是使用数据挖掘和概率来预测结果的过程；例如，可以使用预测模型来确定电子邮件是否是垃圾邮件。对于预测建模，数据科学家使用一个**训练集**，这是一组结果已知的历史数据。这个步骤可以重复更多次，直到模型理解了问题和答案。
*   在**模型评估**阶段，数据科学家可以用两种方式评估模型:坚持和交叉验证。在 Hold-Out 方法中，数据集被分为三个子集:一个**训练集**如我们在建模阶段所说；一个**验证集**，它是一个子集，用于评估在训练阶段建立的模型的性能；**测试集**是一个子集，用于评估模型未来可能的性能。

下面是一个建模和评估的例子:

```
# select dataset and training field
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3" # select field to predictx = np.array(data.drop([predict], 1))
y = np.array(data[predict])# split the dataset into training and test subsets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)linear = linear_model.LinearRegression() # create linear regression modellinear.fit(x_train, y_train) # perform the training of the model
acc = linear.score(x_test, y_test) # calculate the accuracy
print("Accuracy: ", acc) # print the accuracy of the model
```

## 从部署到反馈

数据科学家必须让利益相关者熟悉在不同场景中产生的工具，因此一旦模型被评估并且数据科学家确信它将工作，它就被部署并进行最终测试。

*   **部署**阶段取决于模型的目的，它可能会向有限的一组用户或在测试环境中推广。一个真实的案例研究例子可以是为医疗保健系统设计的模型；该模型可用于一些低风险患者，之后也可用于高风险患者。
*   顾客通常会充分利用**反馈**阶段。部署阶段后的客户可以说模型是否符合他们的目的。数据科学家接受这些反馈，并决定他们是否应该改进模型；这是因为从建模到反馈的过程是高度迭代的。

当模型满足客户的所有要求时，我们的数据科学项目就完成了。

要了解更多，你可以访问我的 [GitHub 库](https://github.com/nlogallo/datascience/tree/master/ds_methodology)，在那里你可以找到一个真实的用例例子等等。

来源: [IBM 数据科学方法论，来自 Coursera](https://www.coursera.org/learn/data-science-methodology?)
书籍来源:[预测未来范例的模型:第 1 卷](https://books.google.it/books?id=OeHFDwAAQBAJ&lpg=SA2-PA62&dq=In%20the%20Data%20Collection%20Stage%2C%20data%20scientists%20identify%20and%20gather%20the%20available%20data%20resources&hl=it&pg=PP1#v=onepage&q&f=false)

农齐奥·洛加洛