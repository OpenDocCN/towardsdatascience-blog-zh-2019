# 如何使用 Python 对 Pandas Dataframe 列进行模糊匹配？

> 原文：<https://towardsdatascience.com/how-to-do-fuzzy-matching-in-python-pandas-dataframe-6ce3025834a6?source=collection_archive---------0----------------------->

![](img/e92d30e7db395c005371a5d56c658494.png)

## 熊猫和 FuzzyWuzzy 的模糊字符串匹配

模糊字符串匹配或搜索是近似匹配特定模式的字符串的过程。这是一个非常受欢迎的 Excel 附件。它给出了一个近似的匹配，并不能保证字符串是精确的，但是，有时字符串会精确地匹配模式。字符串与给定匹配的接近程度由*编辑距离*来衡量。FuzzyWuzzy 使用 [*Levenshtein 距离*](https://en.wikipedia.org/wiki/Levenshtein_distance) 来计算*编辑距离。*

**如何安装 FuzzyWuzzy 包**

要安装 FuzzyWuzzy，您可以使用如下的 *pip* 命令

```
Pip install fuzzywuzzy
Pip install python-Levenshtein
```

人们也可以使用 conda 来安装 FuzzyWuzzy。以下命令将安装该库。

```
**conda install -c conda-forge fuzzywuzzy
conda install -c conda-forge python-levenshtein**
```

现在让我们导入这些包，因为我们已经成功地安装了上面提到的库。

```
from **fuzzywuzzy** import **fuzz**
from **fuzzywuzzy** import **process**
```

现在，让我告诉你如何使用它。

```
In[1]:fuzz.ratio(“Sankarshana Kadambari”,”Sankarsh Kadambari”)
Out[1]:92
In[2]:fuzz.partial_ratio("Sankarshana Kadambari","Sankarsh   Kadambari")
Out[2]:83
```

Fuzz 有各种方法可以比较字符串，比如 ratio()、partial_ratio()、Token_Sort_Ratio()、Token_Set_Ratio()。现在下一个问题是什么时候使用哪个模糊函数。你可以在这里阅读这些场景都有很好的解释。此外，我将在最后提供参考，在那里它将与代码一起被进一步详细解释。

![](img/c403e13c1b6220b3bbe077854aff3b05.png)

# **现在博客的议程如何用这个在熊猫两栏之间的数据框然后导出到 excel？**

```
import pandas as pd 
from fuzzywuzzy import fuzz 
from fuzzywuzzy import processdef checker(wrong_options,correct_options):
    names_array=[]
    ratio_array=[]    
    for wrong_option in wrong_options:
        if wrong_option in correct_options:
           names_array.append(wrong_option)
           ratio_array.append(‘100’)
        else:   
            x=process.
extractOne(wrong_option,correct_options,scorer=fuzz.token_set_ratio)
            names_array.append(x[0])
            ratio_array.append(x[1])
     return names_array,ratio_array
```

在上面的代码片段中，我使用了 token_set_ratio，因为它符合我的要求。我还添加了一个 **if 块**，通过检查第二列中的名称来减少迭代次数，因为有时出现的可能性很大。你可以在 scorer 参数中尝试各种其他方法，我在底部分享了源代码，在那里可以详细研究剩余模糊方法的工作。

我不会说这是实现模糊逻辑的唯一方法，但是您也可以自己尝试如何在减少执行时间的同时增加代码的可伸缩性。

现在让我们传递方法中的参数，并创建一个包含结果的输出 excel 文件。

```
df_Original_List=pd.read_excel(“Original_list.xlsx”,sheet_name=’Sheet1',usecols=[“Original Name Column”])df_To_beMatched=pd.read_excel("To be matched.xlsx",sheet_name='Sheet1',usecols=["To be Matched Column"])
```

为了避免错误，在将列传递给函数之前，清理列是非常重要的。主要的运行时错误是由列中的 NAN 值创建的。我用以下方式处理它们。

```
str2Match = df_To_beMatched[‘To be Matched Column’].fillna(‘######’).tolist()
strOptions =df_Original_List[‘Original Name Column’].fillna(‘######’).tolist()
```

现在让我们传递方法中的参数。

```
name_match,ratio_match=checker(str2Match,strOptions)df1 = pd.DataFrame()
df1[‘old_names’]=pd.Series(str2Match)
df1[‘correct_names’]=pd.Series(name_match)
df1[‘correct_ratio’]=pd.Series(ratio_match)
df1.to_excel(‘matched_names.xlsx’, engine=’xlsxwriter’)
```

瞧，就是这样。我希望这篇文章在你各自的领域对你有用。这是一个在 excel 中使用的非常强大的函数，但现在它也可以在 python 中用于文本分析或分析。

## 您可以在其他资源中找到已实现的剩余方法及其各自的工作方式。

1.  [https://medium . com/@ categitau/fuzzy-string-matching-in-python-68f 240d 910 Fe](https://medium.com/@categitau/fuzzy-string-matching-in-python-68f240d910fe)
2.  https://www.geeksforgeeks.org/fuzzywuzzy-python-library/
3.  [https://www . data camp . com/community/tutorials/fuzzy-string-python](https://www.datacamp.com/community/tutorials/fuzzy-string-python)
4.  [https://towardsdatascience . com/natural-language-processing-for-fuzzy-string-matching-with-python-6632 b 7824 c 49](/natural-language-processing-for-fuzzy-string-matching-with-python-6632b7824c49)
5.  [https://galaxydatatech . com/2017/12/31/fuzzy-string-matching-pandas-fuzzywuzzy/](https://galaxydatatech.com/2017/12/31/fuzzy-string-matching-pandas-fuzzywuzzy/)