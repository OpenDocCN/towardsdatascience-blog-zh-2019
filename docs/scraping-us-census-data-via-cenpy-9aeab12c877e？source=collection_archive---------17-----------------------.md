# 通过 CenPy 的美国人口普查数据

> 原文：<https://towardsdatascience.com/scraping-us-census-data-via-cenpy-9aeab12c877e?source=collection_archive---------17----------------------->

## 用 API 争论人口普查数据

![](img/fe61a32589fe62041739731ee3783dc7.png)

Source: Census.gov (The United States Census logo® is a Federally registered trademark of the U.S. Census Bureau, U.S. Department of Commerce.)

在承担了一个团队项目来检查自然灾害对特定地区的经济影响(通过工资损失)后，我们的假设是使用[季节性自动回归综合移动平均(SARIMA)](https://machinelearningmastery.com/sarima-for-time-series-forecasting-in-python/) 预测来确定没有自然灾害时的历史就业、失业和工资水平。通过将我们的预测与灾后的实际影响进行比较，我们可以估算出该事件的经济影响。

考虑到时间紧迫，我立即惊慌地想到了这样一种可能性，即必须导航一个笨重的 SQL 界面，或者为每个州手动下载不同的 CSV 来获取我需要的信息。此外，我甚至不太确定我最终需要的指标或时间框架。有了这些不确定性，我认为我最好的选择是利用 API 或 webscraper 来获取我需要的数据。然而，我的 API & BeautifulSoup 技能仍然需要一些练习，所以我开始寻找一种替代方法。

幸运的是，Python 社区中充满了富有创造力的开发人员，他们创建了库和包装器来更容易地与 CenPy 这样的数据进行交互。根据创作者的说法，“CenPy ( `sen - pie`)是一个包，它公开了美国人口普查局的 API，并使其易于下载和处理熊猫的人口普查数据。”虽然我将探索我使用的几个特性，但我鼓励大家查看他们的 [GitHub](https://github.com/ljwolf/cenpy) 和[入门笔记本](https://nbviewer.jupyter.org/github/ljwolf/cenpy/blob/master/demo.ipynb)以获取更多信息。

## 入门指南

我在这个练习中使用的唯一进口商品是熊猫和熊猫:

```
import pandas as pd
import cenpy as cen
```

通过调用`explorer.available`，您可以访问`cenpy`知道的所有 API 的标识符列表。幸运的是，我知道我想要访问哪个 API/数据系列，但是如果您想要查看所有可用的 API，您可以通过字典输出或者转换为 Pandas 数据帧来实现，以便于阅读。

```
# Call list of available datasets, verbose = True to include dataset title
datasets = list(cen.explorer.available(verbose=**True**).items())# Convert dictionary to dataframe
datasets = pd.DataFrame(datasets, columns = ['code', 'dataset name'])
```

`code`在这里很重要，因为这将允许您指定您想要建立连接的数据库。`connection`类允许您构造一个查询字符串，并从人口普查服务器发出请求。然后将结果解析成 JSON 并返回。我从[季度工作指标](https://www.census.gov/data/developers/data-sets/qwi.html) (QWI)开始我最初的查询，这是一组 32 个经济指标，包括就业、就业创造/破坏、工资、雇佣和其他就业流动的衡量标准。

```
qwi_connection = cen.base.Connection('QWISA')
```

## 构造查询

根据 Census API 文档，在调用它们的 API 时，有几个参数/变量需求必须使用:端点、指示器、地理和时间。如果没有指定，则采用其他分类变量的默认值。以下是完整基本查询的最低要求示例:

```
api.census.gov/data/timeseries/qwi/sa?get=Emp&for=state:02&year=2012&quarter=1&key=[userkey]
```

我从设置一些默认参数开始:

```
# Specify all counties
g_unit = 'county:*'# Start with one state (chosen arbitrarily) 
g_filter = {'state': '01'}# Specify time period
time = 'from 2003-Q1 to 2018-Q1'# Uses .varslike to pull in all indicator names
cols = qwi_connection.varslike('Emp')    # Employment
hir = qwi_connection.varslike('HirA')    # Hiring
earns = qwi_connection.varslike('Earn')  # Earning
payroll = qwi_connection.varslike('Pay') # Payroll
firm = qwi_connection.varslike('Frm')    # Firm Job Stats
sep = qwi_connection.varslike('sep')     # Seperations # Extend cols to add additional variables
cols.extend(hir)
cols.extend(earns)
cols.extend(payroll)
cols.extend(firm)
cols.extend(sep)
```

## 循环所有状态

我想构建一个循环，允许我遍历所有状态，并将所有结果连接到一个主数据帧中。我首先创建了一个“主”数据帧，可以在其上附加其他状态，而不是创建一个空白数据帧，以确保列顺序匹配，并且新的查询可以适当地连接。

```
# Create the first query / dataframe (with state 01)
master = qwi_connection.query(cols = cols, time = time, geo_filter = g_filter, geo_unit = g_unit)
```

然后，我利用这个[州 FIPS 代码字典](http://code.activestate.com/recipes/577775-state-fips-codes-dict/)创建一个州代码列表，并对其进行迭代:

```
state_codes = {
    'WA': '53', 'DE': '10', 'DC': '11', 'WI': '55', 'WV': '54', 'HI': '15',
    'FL': '12', 'WY': '56', 'NJ': '34', 'NM': '35', 'TX': '48',
    'LA': '22', 'NC': '37', 'ND': '38', 'NE': '31', 'TN': '47', 'NY': '36',
    'PA': '42', 'AK': '02', 'NV': '32', 'NH': '33', 'VA': '51', 'CO': '08',
    'CA': '06', 'AL': '01', 'AR': '05', 'VT': '50', 'IL': '17', 'GA': '13',
    'IN': '18', 'IA': '19', 'MA': '25', 'AZ': '04', 'ID': '16', 'CT': '09',
    'ME': '23', 'MD': '24', 'OK': '40', 'OH': '39', 'UT': '49', 'MO': '29',
    'MN': '27', 'MI': '26', 'RI': '44', 'KS': '20', 'MT': '30', 'MS': '28',
    'SC': '45', 'KY': '21', 'OR': '41', 'SD': '46'
} # Extract numerical state codes
states = list(state_codes.values())
```

最后，我创建了一个`for`循环来迭代所有的州代码，并将结果与我现有的`master`数据帧结合起来:

```
**for** s **in** states:  

    print(f'Scraping **{s}**')

    **try**:
         # Iterate over states 's' g_filter = {'state': s}    

         df = qwi_connection.query(cols=cols, time=time, geo_filer=g_filter, geo_unit = g_unit)                  

        # Concat new df with master df master = pd.concat([master, df]) 

    **except** requests.exceptions.HTTPError:
         **pass**
```

请注意，您可以在单个 API 查询中包含多达 50 个变量，并且每天可以对每个 IP 地址进行多达 500 次查询。每天每个 IP 地址超过 500 次查询需要您注册一个普查密钥。该密钥将是您在建立`connection`时指定的数据请求 URL 字符串的一部分。

我发现我对所有州的所有县的查询超过了总的 IP 限制，即使在使用了我的 API 密钥之后也是如此，并且它使我在系统之外超时。这迫使我在两天内将我的查询分成两部分。

我希望这是对你有帮助的入门，并且你能够探索 CenPy 的所有可能性！