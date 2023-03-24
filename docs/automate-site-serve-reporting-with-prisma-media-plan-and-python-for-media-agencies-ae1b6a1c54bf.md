# 使用 Prisma Media Plan 和 Python 为媒体机构实现网站服务报告自动化

> 原文：<https://towardsdatascience.com/automate-site-serve-reporting-with-prisma-media-plan-and-python-for-media-agencies-ae1b6a1c54bf?source=collection_archive---------12----------------------->

![](img/9e3808cc6bb7375c0130a70c6d31b3fb.png)

Photo by [Franck V.](https://unsplash.com/@franckinjapan?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

为了拥有有效的媒体，媒体计划必须利用多种分销渠道(即数字显示、电视、 [OOH](https://en.wikipedia.org/wiki/Out-of-home_advertising) )。有研究表明，开展[跨渠道营销活动](https://econsultancy.com/what-is-cross-channel-marketing-and-why-do-you-need-it/)可以显著提升业绩。这些营销计划的一个关键组成部分是使用出版商自己跟踪广告，而不是通过广告服务器。在行业中，由发布者手动跟踪的广告被标记为“站点服务”。在这些情况下，代理商依赖于出版商提供与这些广告相关的相关数据(印象、点击等)。

从我的经验来看，从出版商收集这些数据的过程通常遵循这些特定的步骤:为特定的出版商制定网站服务布局的子集媒体计划，创建具有所有所需指标(印象、视频观看等)的 excel 报告模板，将 excel 模板发送给出版商填写，出版商发回数据。传统上，这是一个高度手工的过程，通常由媒体购买者自己进行。本文的范围集中在将模板发送给出版商之前的所有步骤的自动化上。我们将使用 python 作为我们的首选语言。这是自动化站点服务报告系列的第 1 部分，共 2 部分。第 2 部分将介绍如何在电子邮件中发送模板。

我们从导入这个项目所需的包开始。由于大多数 excel 电子表格的表格性质，在我们的例子中是媒体计划，Pandas 包的 DataFrame 将是在 python 环境中操作数据的绝佳选择。xlsxwriter 是一个包，它允许我们将数据帧写入 excel 工作表。虽然我们可以通过简单的 Pandas 来实现，但是 xlsxwriter 允许更深入的 excel 操作。

```
import pandas as pd 
import xlsxwriter
```

我还假设你的媒体计划有某种级别的清理数据(放置级别的粒度，如果需要，请阅读这篇[文章](https://medium.com/@harrychaw/cleaning-prisma-media-plan-with-python-for-analytics-e67f850730f)作为参考)。

```
#Read in Prisma Media Plan file
mp = pd.read_excel(‘Media Plan File’)"""Create day level of granularity"""#Create date range of placements 
mp[‘Date Range’] = (pd.to_datetime(mp[‘End Date’]) — pd.to_datetime(mp[‘Start Date’])).dt.days#Use Date Range to repeat placement 
mp = mp.reindex(mp.index.repeat(mp[‘Date Range’]))#Filter out placements that are site served
mp = mp['Placement Name'].str.contains('_SiteServed_')
```

在对媒体计划进行一些简单的操作后，我们需要按出版商对媒体计划进行分组，因为每个出版商只负责他们的广告投放。在那里，我们使用列表理解来划分出版者和他们各自的内容。

```
SiteGroup = mp.groupby(‘Site’)vendorList= [contents for vendor, contents in SiteGroup]
```

计算机编程最好的方面之一是能够有效地循环函数，利用计算机的能力来做它们最擅长的事情:计算。因此，在本例中，我们将遍历 vendorList 列表，以便为每个供应商创建模板。

```
for vendor in vendorList:
 #Creating the export excelName; setting by the first Campaign name     and Supplier value
```

请注意，以下所有代码都将嵌套在前一个循环中。在这一步中，我们将从我们的 *vendorList* 列表中获取活动和出版商的名称，用于命名我们的 excel 模板文件。

```
campaignName = vendor[‘Campaign Name’].iloc[0]
supplierName = vendor[‘Site’].iloc[0]
string = (‘{0}_{1}’+’.xlsx’).format(campaignName, supplierName)
```

现在我们开始实际创建 excel 模板。我们首先调用一个带有熊猫数据帧的实例，用 xlsxwriter 作为引擎。然后，我们从 *vendorList* 列表中取出我们的供应商变量，并将该数据插入到我们的 *writer* 实例中，从而用数据填充 excel 表格。之后，我们创建一个*工作簿*和*工作表*实例来与 excel 工作表交互。我们将 excel 表格命名为“站点”。

```
writer = pd.ExcelWriter(string, engine=’xlsxwriter’) 
vendor.to_excel(writer, sheet_name=’SITE’,index=False)
workbook = writer.book
worksheet = writer.sheets[‘SITE’]
```

与发布者处理网站服务报告的一个关键问题是，他们有时会修改我们提供的模板，以包含其他可能有帮助的指标。然而，如果由于某种原因这造成了障碍，我们可以应用一个宏来防止对我们的 excel 模板的任何修改。这是通过我们附加到工作表实例上的 protect 属性实现的。但是因为我们确实需要发布者在 excel 模板中输入数据，所以我们创建了一个允许解锁某些列的格式，从而允许输入数据。

```
worksheet.protect()
unlocked = workbook.add_format({‘locked’: 0})
```

在我们修改的下一段代码中，将出现在模板中的列。我们创建了文本换行格式，以确保单元格中的所有文本都能正确显示。然后，我们为发布者输入数据的地方标记四个列标题。然后我们解锁这些列，以允许使用我们之前创建的*解锁*格式输入数据。

```
text_format = workbook.add_format({‘text_wrap’: True})worksheet.write(‘F1’,’Publisher Impressions’)
worksheet.write(‘G1’,’Publisher Clicks’)
worksheet.write(‘H1’,’Video: Starts’)
worksheet.write(‘L1’,’Video: 100% Complete’)#where the vendors will input data
worksheet.set_column(‘F2:O2’,25,unlocked) 
```

在电子表格中看到任何文本被未展开的单元格截断，这可能是 MS excel 中最难看的事情之一。因此，在接下来的几行中，我们调整列以显示适当的长度来容纳全文。

```
worksheet.set_column(‘A:B’,vendor[‘Campaign Name’].map(len).max(),text_format)
worksheet.set_column(‘C:C’,vendor[‘Placement’].map(len).max(),text_format)worksheet.set_column(‘D:D’,15)
worksheet.set_column(‘E:Z’,25)
```

我们最后关闭工作簿并保存 excel 模板。

```
workbook.close()
writer.save()
```

完整代码见下文

回顾一下，我们已经编写了一个 python 脚本，解析出媒体计划中的现场投放，并为发布者创建一个 excel 模板，然后填写以满足报告需求。在项目实施过程中可能出现的问题包括广告位的命名惯例、广告位的突然变更但未反映在媒体计划中，以及许多其他微小的变化。

该项目的下一步将是自动化与发送 excel 模板给出版商相关的电子邮件。

在一个理想的世界(以媒体代理为中心的世界)，网站服务将被取消，所有的交易将通过你选择的广告服务器(可能是 DCM，实际上最有可能是 DCM…)完成。然而，在我看来，下一个最好的事情是在出版商之间统一一个 API，我们可以手动调用他们的服务器来获取印象和其他指标。但目前我们所能做的最好的事情是尽量减少错误，并使出版商和代理商之间的数据移交过程标准化。

如有疑问、意见或担忧(或突出的问题)，请通过 LinkedIn[https://www.linkedin.com/in/harrychaw/](https://www.linkedin.com/in/harrychaw/)联系我