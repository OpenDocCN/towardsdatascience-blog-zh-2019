# Python 中加速数据科学发展的 8 个技巧

> 原文：<https://towardsdatascience.com/8-tricks-in-python-to-accelerate-your-data-science-development-cad3b634e140?source=collection_archive---------8----------------------->

让我们简短，让我们有效。少字多码(*多模因图解*)。这就是你要找的，对吗？；)

![](img/8601e288258ece3f7b7b2c9f5eb5c6e7.png)

# Python 语言技巧

## 检查一行代码中的条件(断言)

![](img/7378f0ee5071230d7cc788e2ac9043d6.png)

例如，我们将检查训练和测试函数是否具有相同的列(当您预测测试数据集并且没有应用相同的功能工程时，这是一个常见的错误:)

常见 Python:

```
label = df_train['label_column']
df_train = df_train.drop('label_column',axis=1)
if df_train.column.tolist() != df_test.column.tolist():
   print('Train and test column do not match')
   print(sys.exit())
```

Python 技巧:

```
assert df_train.column.tolist() == df_test.column.tolist(),'Train and test column do not match'
```

## 在一行代码中循环(单行代码)

![](img/f4892c99b92d9c01677514f8a86c8fee.png)

例如，我们将填充一个从 0 到 100 的偶数列表，并为 ood 编号填充 0(即[0,0,2,0,4,0…])

常见 Python:

```
my_awesome_list = []
for i in range [0,100]:
    if i%2 = 0:
        my_awesome_list.append(i)
    else:
        my_awesome_list.append(0)**my_awesome_list = [0,0,2,0,3,4]**
```

Python 技巧:

```
my_awesome_list = [i if i%2==0 else 0 for i in range(0,100) ]**my_awesome_list = [0,0,2,0,3,4]**
```

## 创建函数而不创建函数(lambda)

![](img/b973b8db2178d97b2d7b7daac35f4734.png)

计算列表中每个元素的双精度值:

常见 Python:

```
a = [2,2,2]
for i in range(0,len(a)):
    a[i] = a[i]*2**[4, 4, 4]**
```

Python 技巧:

```
a = [2,2,2]
double = lambda x : x*2
a = [double(a_) for a_ in a]**[4, 4, 4]**
```

## 使用另一个列表转换一个列表，不使用循环(map)

![](img/5756d9f517ed699d8bb7b18b4c53ff20.png)

在一个完全否定列表中转换一个数字列表。

常见 Python:

```
a = [1,1,1,1]
for i in range(0,len(a)):
    a[i] = -1 * a[i]**[-1, -1, -1, -1]**
```

Python 技巧:

```
a = [1,1,1,1]
a = list(map(lambda x: -1*x, a))**[-1, -1, -1, -1]**
```

## 乍一看不要滥用简单语法(try: except 和 global)

![](img/15741160fe4c258a4488f591b5f39866.png)

我在初级 Python 开发人员的 lof 代码中注意到的一个常见错误是使用太多 try/except 和全局语法来开发代码，例如:

```
def compare_two_list(list_a,list_b):
    global len_of_all_list
    try:
        for i in range(0,len_of_all_list):
            if list_a[i] != list_b[i]:
                print('error')
    except:
        print('error')global len_of_all_listlen_of_all_list = 100
list_a = [1]*len_of_all_list
len_of_all_list = len_of_all_list+1
list_b = [1]*len_of_all_listcompare_two_list(list_a,list_b)**'error'**
```

调试这样的代码真的很复杂。使用 try:除非您需要通过处理特定的异常来操作代码中的某些内容。例如，这里使用“try: except”语句来处理我们的问题的一种方法是忽略 list_a 的最大索引之后的所有索引:

```
def compare_two_list(list_a,list_b):
    global len_of_all_list
    try:
        for i in range(0,len_of_all_list):
            if list_a[i] != list_b[i]:
                print('error')
    except IndexError:
        print('It seems that the two lists are different sizes. They was similar until index {0}'.format(len_of_all_list-1))
        return

global len_of_all_list
len_of_all_list=100
list_a = [1]*len_of_all_list
len_of_all_list = len_of_all_list+1
list_b = [1]*len_of_all_list
compare_two_list(list_a,list_b)**It seems that the two lists are different sizes. They was similar until index 101.**
```

# 数据科学图书馆技巧

## 基于条件(np.where())填充 DataFrame 列

![](img/d32df3044ab6e2a17fcb9ee8d0af5e8e.png)

作为示例，我们将基于一个条件来填充一列，如果人数等于 1，则意味着该人是独自一人，否则该人不是独自一人(为了简化，我们假设 0 不存在于“人数”列中):

常见 Python:

```
df['alone'] = ''
df.loc[df['number_of_persons'] == 1]]['alone'] = 'Yes'
df.loc[['number_of_persons'] != 1]]['alone'] = 'No'
```

Python 技巧:

```
df['alone'] = np.where(df['number_of_persons']==1,'Yes','No')
```

## 获取所有数字列(pd。DF()。select_dtypes(include=[]))

![](img/0f18173aedeb9b9b3e94ca274c4759ae.png)

对于大多数机器学习算法，我们需要给它们数值。Pandas DataFrame 提供了一种选择这些列的简单方法。您还可以使用 select_dtypes 选择任何类型的数据类型作为对象、分类...

常见 Python:

```
df_train.info()**<class 'pandas.core.frame.DataFrame'>
Index: 819 entries, 0_0 to 2_29
Data columns (total 4 columns):
image_id                     812 non-null object
age                          816 non-null int64
gender                       819 non-null object
number_of_persons            734 non-null float64
dtypes: float64(10), object(3)
memory usage: 89.6+ KB** numerical_column = ['age','number_of_persons']
X_train = df_train[numerical_column]
```

Python 技巧:

```
X_train = df_train.select_dtypes(include=['int64','float64'])
```

## 获取 DataFrame 选择中条件的倒数(。[~条件])

![](img/90103fa570b1e08df6f596486b2e5107.png)

例如，我们将为成年人和未成年人创建两个列。

常见 Python:

```
minor_check = df.age.isin(list(range(0,18)))
df['minor'] = df[minor_check]df['adult’] = df[not (minor_check)]**ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().**
```

Python 大师:

我们知道。isna()和 isin()可以检索 nan 值和 dataframe 列中的列表值。但是 isNOTna() ans isNOTin()不存在，符号来了~(也可以用 np.invert():)。

```
minor_check = df.age.isin(list(range(0,18)))
df['minor'] = df[minor_check]
df['adult’] = df[~minor_check]
```

希望这些技巧会有所帮助，请不要犹豫，分享你的！:)