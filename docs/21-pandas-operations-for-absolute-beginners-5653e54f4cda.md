# 绝对初学者的 21 熊猫操作

> 原文：<https://towardsdatascience.com/21-pandas-operations-for-absolute-beginners-5653e54f4cda?source=collection_archive---------2----------------------->

## 先决条件:Python 和 NumPy 基础。

![](img/459c93e0f221631b71b0c24076f49446.png)

Source: [https://pandas.pydata.org/](https://pandas.pydata.org/)

## 介绍

Pandas 是一个易于使用并且非常强大的数据分析库。像 NumPy 一样，它将大多数基本操作矢量化，即使在 CPU 上也可以并行计算，从而提高计算速度。这里指定的操作是非常基本的，但是如果你刚刚开始接触熊猫，那么这些操作就太重要了。您需要将 pandas 作为“pd”导入，然后使用“pd”对象来执行其他基本的 pandas 操作。

1.  **如何从 CSV 文件或文本文件中读取数据？**

CSV 文件以逗号分隔，因此为了读取 CSV 文件，请执行以下操作:

```
df = pd.read_csv(file_path, sep=’,’, header = 0, index_col=False,names=None)**Explanation:**‘read_csv’ function has a plethora of parameters and I have specified only a few, ones that you may use most often. A few key points:a) **header=0** means you have the names of columns in the first row in the file and if you don’t you will have to specify header=Noneb) **index_col = False** means to not use the first column of the data as an index in the data frame, you might want to set it to true if the first column is really an index.c) **names = None** implies you are not specifying the column names and want it to be inferred from csv file, which means that your header = some_number contains column names. Otherwise, you can specify the names in here in the same order as you have the data in the csv file. 
**If you are reading a text file separated by space or tab, you could simply change the sep to be:**sep = " " or sep='\t'
```

**2。如何使用预先存在的列或 NumPy 2D 数组的字典创建数据框？**

使用字典

```
# c1, c2, c3, c4 are column names. 
d_dic ={'first_col_name':c1,'second_col_names':c2,'3rd_col_name':c3} df = pd.DataFrame(data = d_dic)
```

使用 NumPy 数组

```
np_data = np.zeros((no_of_samples,no_of_features)) #any_numpy_arraydf = pd.DataFrame(data=np_data, columns = list_of_Col_names)
```

**3。如何可视化数据框中顶部和底部的 x 值？**

```
df.head(num_of_rows_to_view) #top_values
df.tail(num_of_rows_to_view) #bottom_valuescol = list_of_columns_to_view df[col].head(num_of_rows_to_view)
df[col].tail(num_of_rows_to_view)
```

**4。如何重命名一列或多列？**

```
df = pd.DataFrame(data={'a':[1,2,3,4,5],'b':[0,1,5,10,15]})new_df = df.rename({'a':'new_a','b':'new_b'})
```

重要的是将返回的数据帧存储到新的数据帧#中，因为重命名不是原位的。

**5。如何获取列表中的列名？**

```
df.columns.tolist()
```

如果您只想遍历名称，不使用 tolist()函数也可以做到这一点，但它会将所有内容作为索引对象返回。

**6。如何求一个数列中值的出现频率？**

```
df[col].value_counts() #returns a mapper of key,frequency pairdf[col].value_counts()[key] to get frequency of a key value
```

**7。如何重置现有列或其他列表或数组的索引？**

```
new_df = df.reset_index(drop=True,inplace=False)
```

如果你做 **inplace=True** ，就没有必要把它存储到 new_df。此外，当您将索引重置为 pandas RangeIndex()时，您可以选择保留旧索引或使用' drop '参数将其删除。您可能希望保留它，特别是当它最初是列之一，并且您临时将其设置为 newindex 时。

**8。如何删除列？**

```
df.drop(columns = list_of_cols_to_drop)
```

**9。如何更改数据框中的索引？**

```
df.set_index(col_name,inplace=True)
```

这会将 col_name col 设置为索引。您可以传递多个列来将它们设置为索引。inplace 关键字的作用和以前一样。

10。如果行或列有 nan 值，如何删除它们？

```
df.dropna(axis=0,inplace=True)
```

axis= 0 将删除任何具有 nan 值的列，这可能是您在大多数情况下不需要的。axis = 1 将只删除任何列中具有 nan 值的行。原地同上。

11。在给定条件下，如何对数据帧进行切片？

您总是需要以逻辑条件的形式指定一个掩码。
例如，如果您有年龄列，并且您想要选择年龄列具有特定值或位于列表中的数据框。然后，您可以实现如下切片:

```
mask = df['age'] == age_value 
or
mask = df['age].isin(list_of_age_values)result = df[mask]
```

具有多个条件:例如，选择高度和年龄都对应于特定值的行。

```
mask = (df['age']==age_value) & (df['height'] == height_value)result = df[mask]
```

12。给定列名或行索引值，如何对数据框进行切片？

这里有 4 个选项:at、iat、loc 和 iloc。其中“iat”和“iloc”在提供基于整数的索引的意义上是相似的，而“loc”和“at”提供基于名称的索引。

这里要注意的另一件事是“iat”，在为单个元素“提供”索引时，使用“loc”和“iloc”可以对多个元素进行切片。

```
Examples:a) 
df.iat[1,2] provides the element at 1th row and 2nd column. Here it's important to note that number 1 doesn't correspond to 1 in index column of dataframe. It's totally possible that index in df does not have 1 at all. It's like python array indexing.b)
df.at[first,col_name] provides the value in the row where index value is first and column name is col_namec)
df.loc[list_of_indices,list_of_cols] 
eg df.loc[[4,5],['age','height']]
Slices dataframe for matching indices and column namesd)
df.iloc[[0,1],[5,6]] used for interger based indexing will return 0 and 1st row for 5th and 6th column.
```

13。如何对行进行迭代？

```
iterrows() and itertuples()for i,row in df.iterrows():
    sum+=row['hieght']iterrows() passess an iterators over rows which are returned as series. If a change is made to any of the data element of a row, it may reflect upon the dataframe as it does not return a copy of rows.itertuples() returns named tuplesfor row in df.itertuples():
    print(row.age)
```

**14。如何按列排序？**

```
df.sort_values(by = list_of_cols,ascending=True) 
```

**15。如何将一个函数应用到一个系列的每个元素上？**

```
df['series_name'].apply(f) where f is the function you want to apply to each element of the series. If you also want to pass arguments to the custom function, you could modify it like this.def f(x,**kwargs):
    #do_somthing
    return value_to_storedf['series_name'].apply(f, a= 1, b=2,c =3)If you want to apply a function to more than a series, then:def f(row):
    age = row['age']
    height = row['height']df[['age','height']].apply(f,axis=1)
If you don't use axis=1, f will be applied to each element of both the series. axis=1 helps to pass age and height of each row for any manipulation you want.
```

16。如何将函数应用于数据框中的所有元素？

```
new_df = df.applymap(f)
```

**17。如果一个序列的值位于一个列表中，如何对数据帧进行切片？**

使用掩码和 isin。要选择年龄在列表中的数据样本:

```
df[df['age'].isin(age_list)]
```

要选择相反的情况，年龄不在列表中的数据样本使用:

```
df[~df['age'].isin(age_list)]
```

**18。如何按列值分组并在另一列上聚合或对其应用函数？**

```
df.groupby(['age']).agg({'height':'mean'})
```

这将按系列“年龄”对数据框进行分组，对于高度列，将应用分组值的平均值。有时，您可能希望按某一列进行分组，并将其他列的所有相应分组元素转换为一个列表。您可以通过以下方式实现这一目标:

```
df.groupby(['age']).agg(list)
```

**19。如何为特定列的列表中的每个元素创建其他列的副本？**

这个问题可能有点混乱。我实际上的意思是，假设你有下面的数据帧 df:

```
Age Height(in cm)20  180
20  175
18  165
18  163
16  170
```

在使用列表聚合器应用 group-by 后，您可能会得到如下结果:

```
Age Height(in cm)
20  [180,175]
18  [165,163]
16  [170]
```

现在，如果您想通过撤消上一次操作返回到原始数据框，该怎么办呢？你可以使用熊猫 0.25 版本中新引入的名为 explode 的操作来实现。

```
df['height'].explode() will give the desired outcome.
```

**20。如何连接两个数据帧？**

假设您有两个数据帧 df1 和 df2，它们具有给定的列名、年龄和高度，并且您想要实现这两个列的连接。轴=0 是垂直轴。这里，结果数据帧将具有从数据帧追加的列:

```
df1 --> name,age,height
df2---> name,age,heightresult = pd.concat([df1,df2],axis=0)
```

对于水平连接，

```
df1--> name,agedf2--->height,salaryresult = pd.concat([df1,df2], axis=1) 
```

**21。如何合并两个数据框？**

```
For the previous example, assume you have an employee database forming two dataframes likedf1--> name, age, heightdf2---> name, salary, pincode, sick_leaves_takenYou may want to combine these two dataframe such that each row has all details of an employee. In order to acheive this, you would have to perform a merge operation.df1.merge(df2, on=['name'],how='inner')This operation will provide a dataframe where each row will comprise of name, age, height, salary, pincode, sick_leaves_taken. how = 'inner' means include the row in result if there is a matching name in both the data frames. For more read: [https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html?highlight=merge#pandas.DataFrame.merg](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html?highlight=merge#pandas.DataFrame.merge)e
```

## 总结

作为初学者，对于任何数据分析项目，您可能需要非常了解这些操作。我一直认为 Pandas 是一个非常有用的库，现在你可以集成各种其他数据分析工具和语言。在学习支持分布式算法的语言时，了解 pandas 的操作甚至会有所帮助。

## 接触

如果你喜欢这篇文章，请鼓掌并与其他可能会发现它有用的人分享。我真的很喜欢数据科学，如果你也对它感兴趣，让我们在 LinkedIn[上联系或者在这里关注我走向数据科学平台。](https://www.linkedin.com/in/parijatbhatt/)