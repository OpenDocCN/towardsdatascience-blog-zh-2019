# 使用 xmltree 将多层 xml 文件转换为 python 中的数据帧

> 原文：<https://towardsdatascience.com/converting-multi-layered-xml-files-to-dataframes-in-python-using-xmltree-a13f9b043b48?source=collection_archive---------2----------------------->

![](img/76fc64157d08b49eb8dceb77c13e5775.png)

Relativity by MC Esher ( 2015 The M.C. Esher Company- Baarn, Netherlands)

我发现使用任何有效的编程语言的最大优点之一是，这种语言有助于将抽象的数据结构分解成简单的数据点，程序员可以将这些数据点转换成任何方便的结构。数据结构仅仅是一种形式，它表现了一种不变的本质的可能排列。因此，当您作为程序员查看 excel 文件时，您会意识到 excel 仅仅是数据的“形式”,它呈现了数据“本质”的一种可能结构，而数据“本质”仅仅是点。

允许大型数据集以多层格式呈现的数据结构之一是可扩展标记语言，或者更通俗地称为 XML。本质上，xml 是一种标记语言，使数据对人和机器都可读。不用说，这带来了巨大的优势。这也允许在一个数据集内嵌套变量，即以层的形式呈现数据。例如，一个 xml 文档看起来像下面这样，

```
#A sample xml file <?xml version="1.0" encoding="UTF-8"?><scenario><world><region name="USA"><AgSupplySector name="Corn"><AgSupplySubsector name="Corn_NelsonR"><AgProductionTechnology name="Corn_NelsonR_IRR_hi"><period year="2015"><agProdChange>0.0043</agProdChange></period></AgProductionTechnology>
</AgSupplySubsector>
</AgSupplySector>
</region>
</world>
</scenario>
```

因此，您可以看到数据的各种属性以层的形式呈现。然而，使这些数据可读可能是一个挑战，尤其是在处理巨大的 xml 文件时。

**问题-** 在本文中，我展示了一个易于修改的 python 脚本，它解析一个包含 6 层的 xml，并在一个简单的数据帧中呈现数据，非常适合分析。本例中使用的 xml 文件是一个数据集，按国家、 ***、作物、流域、技术、年份和气体类型*** 显示温室气体排放量。在这个例子中，我使用了 python 中的 xmltree 包。样本数据集“ag_prodchange_ref_IRR_MGMT.xml”是 GCAM 模型数据系统的一部分，可在此处[访问](https://github.com/JGCRI/gcamdata)。该文件包含 31 个地区、大约 2000 个盆地、12 种作物类型和 15 种气体类型的数据。

我们希望将 xml 文件转换成如下所示的简单格式，

![](img/3bc28c35ca138b884a2384bd8237f52f.png)

因此，解决这个问题最简单的方法是考虑编写一个分层的 for 循环来解析 XML 以接收输出，然后在 for 循环的最后一部分将数据转储到 dataframe 中。那么，我们开始吧。

首先，让我们导入包并使用 xmltree 读入数据。我们也将需要熊猫，因为我们将与数据帧。您将看到，我们最初在 xml 树中使用 parse 函数解析 xml 对象，然后将整个树转储到一个名为 root 的变量中。

```
**import** xml.etree.cElementTree **as** et
**import** pandas **as** pdtree=et.parse(**'all_aglu_emissions.xml'**)
root=tree.getroot()
```

现在，让我们为我们的层创建一些空列表。如上所述，有 6 层需要考虑，因此我们将创建对应于每层的空列表。

```
Year=[]
Gas=[]
Value=[]
Technology=[]
Crop=[]
Country=[]
```

让 for 循环开始！可以把它想象成一棵决策树，在这里你想要得到一个特定的值。所以，你要在特定地区、特定供应部门、特定技术类型、特定年份的天然气中寻找特定价值。下图描述了 for 循环。

![](img/3ca0a73c19bfbe2b5f70ec7a4c90fa91.png)

A simple visual representation of the for loop we will write out in the code.

xml 树方便地允许用户通过迭代来解析元素。现在，具有挑战性的部分是创建一个内部存储器，以便 loop 记住每个部分的内容。否则，我们可能会陷入阿里阿德涅在《盗梦空间》中生动描述的境地，

等等，我们到底要进入谁的潜意识呢

![](img/0d0615117e67c183c05912d0e69217fb.png)

Much like this scene in inception the for loop can mess up the files python remembers at each step

因此，在循环的每个阶段，我们都希望将迭代保存到一个变量中，这个变量将成为下一阶段循环的基础。我们的第一阶段是“区域”。我们将把这个阶段的结果保存到一个名为“根 1”的变量中。然后，我们将在第二阶段迭代这个新变量，并将结果保存到另一个名为“root 2”的临时变量中。因此，前两个级别的代码如下所示，

```
**for** reg **in** root.iter(**'region'**):
    root1=et.Element(**'root'**)
    root1=reg
    **for** supply **in** root1.iter(**'AgSupplySector'**):
        root2=et.Element(**'root'**)
        root2=(supply)
```

注意 et。元素(' root ')创建一个空的 xml 对象来存储我们的结果。另外，注意你声明的 for 变量很重要，所以记住上面的' x '和' reg'。我们稍后将需要它们来提取我们的属性。所以，把这个结构再扩展 4 个层次，

```
**for** x **in** root.iter(**'region'**):
    root1=et.Element(**'root'**)
    root1=x
    **for** supply **in** root1.iter(**'AgSupplySector'**):
        root2=et.Element(**'root'**)
        print(supply)
        root2=(supply)
        **for** tech **in** root2.iter(**'AgProductionTechnology'**):
            root3 = et.Element(**'root'**)
            root3=(tech)
            **for** yr **in** root3.iter(**'period'**):
                root4 = et.Element(**'root'**)
                root4=yr
                **for** gas **in** root4.iter(**'Non-CO2'**):
                    root5 = et.Element(**'root'**)
                    root5=gas
```

现在，对于最后一个级别，我们希望将实际值存储在 xml 文件中。这些可以在 xmltree 中使用提取特定属性的“attrib”函数提取。所以，回到我们的 xml 结构，每一层都包含特定的属性。例如，在下面的行中，区域部分包含一个名为“name”的属性，该属性被设置为 USA。

```
<region name="USA">
```

因此，为了提取名称，我们将编写以下代码行，

```
reg.attrib['name']
```

因此，按照这个逻辑，让我们从上面提取所有属性，并将它们分配给我们在第一步中生成的空列表。所以，代码变成了，

```
Technology.append(tech.attrib[**'name'**])
Value.append(em.text)
Gas.append(gas.attrib[**'name'**])
Year.append(yr.attrib[**'year'**])
Crop.append(supply.attrib[**'name'**])
Country.append(x.attrib[**'name'**])
```

请注意，该值不像其他列表那样具有属性函数。我们直接调用了一个名为‘text’的对象。这是因为该值存储在没有特定属性的图层中。所以我们可以直接访问对象。

```
<input-emissions>0.0043</input-emissions>
```

所以，我们上面的代码就变成了，

```
**for** x **in** root.iter(**'region'**):
    root1=et.Element(**'root'**)
    root1=x
    **for** supply **in** root1.iter(**'AgSupplySector'**):
        root2=et.Element(**'root'**)
        print(supply)
        root2=(supply)
        **for** tech **in** root2.iter(**'AgProductionTechnology'**):
            root3 = et.Element(**'root'**)
            root3=(tech)
            **for** yr **in** root3.iter(**'period'**):
                root4 = et.Element(**'root'**)
                root4=yr
                **for** gas **in** root4.iter(**'Non-CO2'**):
                    root5 = et.Element(**'root'**)
                    root5=gas
                    **for** em **in** root5.iter(**'input-emissions'**):
                        Technology.append(tech.attrib[**'name'**])
                        Value.append(em.text)
                        Gas.append(gas.attrib[**'name'**])
                        Year.append(yr.attrib[**'year'**])
                        Crop.append(supply.attrib[**'name'**])
                        Country.append(x.attrib[**'name'**])
```

现在，简单的部分。让我们使用填充的列表在 pandas 中创建一个数据帧。

```
df = pd.DataFrame({**'Year'**: Year,**'Gas'**:Gas,**'Value'**:Value,**'Technology'**:Technology,**'Country'**:Country,**'Crop'**:Crop })
```

如果我们打印出这个数据帧的头部，我们会得到，

```
Year        Gas         Value    Technology Country  Crop
0  1975  SO2_1_AWB   3.98749e-05  Corn_NelsonR     USA  Corn
1  1975    NOx_AWB   0.000285263  Corn_NelsonR     USA  Corn
2  1975     CO_AWB  0.0089533275  Corn_NelsonR     USA  Corn
3  1975  NMVOC_AWB  0.0005259348  Corn_NelsonR     USA  Corn
4  1975    CH4_AWB  0.0002027923  Corn_NelsonR     USA  Corn
```

这就对了。简单回顾一下，在将 XML 转换为 dataframe 时，需要记住一些原则，

1.  永远记住 xml 本身的层次，包括每一层中的属性
2.  编写 for 循环时，确保在每个阶段都将解析的数据保存到临时变量中。
3.  在循环的最后一步将数据写到空列表中。
4.  请记住，结构就是数据的形式。你在修改形式的同时保持了它的本质。

以上代码可以在这里找到-[https://github.com/kanishkan91/ConvertXMLtoDataframepy](https://github.com/kanishkan91/ConvertXMLtoDataframepy)

上例中使用的基本 xml 可以在这里找到-[https://github . com/kanishkan 91/ConvertXMLtoDataframepy/blob/master/all _ aglu _ emissions . XML](https://github.com/kanishkan91/ConvertXMLtoDataframepy/blob/master/all_aglu_emissions.xml)

一如既往，欢迎任何反馈。