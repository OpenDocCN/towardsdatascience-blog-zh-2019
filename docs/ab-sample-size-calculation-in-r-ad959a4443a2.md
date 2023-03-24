# R 中 AB 样本量的计算

> 原文：<https://towardsdatascience.com/ab-sample-size-calculation-in-r-ad959a4443a2?source=collection_archive---------21----------------------->

## R 中计算样本容量和实验持续时间的一些有用工具

![](img/6e2ec34bc1572f1e02dc82c8187c9ed6.png)

## 介绍

为利益相关者执行即席分析可能非常耗时。此外，我还经常被问到一些问题。所以我花了一些时间开发一些工具，让我的“非技术”同事在 r 中使用。

最常见的问题之一是“我需要多大的样本才能达到显著性？”，后面常常是“我需要运行我的实验多长时间？”。为此，我开发了一些简单的代码，供人们在需要回答这些问题时使用。所有用户需要做的是传递一些基线数字到我创建的一些函数中，他们可以确定他们的样本大小要求和实验持续时间。

## 样本量、统计功效和实验持续时间

幸运的是，通过了解一些简单的信息，R 中的 *pwr()* 包可以相当轻松地回答这两个问题。 *Pwr()* 帮助您在进行实验之前进行功效分析，使您能够确定每个实验条件下的样本量。

计算功耗分析所需的**四个量**有着密切的关系，如果我们有剩余的输入，我们就能够计算出其中的任何一个值:

> *1。样本量(n)*
> 
> *2。效果尺寸*
> 
> *3。显著性水平(α)= P(I 型误差)=发现不存在的影响的概率*
> 
> *4。功效= 1 — P(第二类误差)=发现存在效应的概率*

由于您的显著性水平(3)和功效(4)通常是固定值，只要您可以输入对照和变量的效应大小(2)，您就可以确定所需的样本大小(1)。

幸运的是， *pwr()* 包中的 *ES.h()* 函数计算出我们的效应大小，以便我们进行功耗分析。我们通常会知道我们的控制条件的当前转换率/性能，但是变量的影响根据定义几乎是未知的。然而，我们可以计算一个预期的影响大小，给定一个期望的提升。一旦计算出这些影响，它们将被传递到 *pwr.p.test()* 函数中，该函数将计算我们的样本大小，前提是 *n* 留空。为了使这种分析对用户友好，我将前面提到的两个函数打包成一个新函数，名为*sample _ size _ calculator()*。

此外，因为我们将使用这些信息来计算运行实验所需的天数，所以我也创建了一个 *days_calculator()* 函数，它将使用我们的样本大小计算的输出:

```
sample_size_calculator <- function(control, uplift){
variant <- (uplift + 1) * control
baseline <- ES.h(control, variant)
sample_size_output <- pwr.p.test(h = baseline,
n = ,
sig.level = 0.05,
power = 0.8)
if(variant >= 0)
{return(sample_size_output)}
else
{paste("N/A")}
}days_calculator <- function(sample_size_output, average_daily_traffic){
days_required <- c(sample_size_output * 2)/(average_daily_traffic)
if(days_required >= 0)
{paste("It will take this many days to reach significance with your current traffic:", round(days_required, digits = 0))}
else
{paste("N/A")}
}
```

如果您正在使用此工具，您只需指定您的控制转换率和所需的提升:

```
control <- 0.034567uplift <- 0.01
```

并运行 *sample_size_calculator()* 函数:

```
sample_size_calculator(control, uplift)sample_size_output <- sample_size_output$nsample_size_output
```

然后，在给定这些值的情况下，您将获得所需的样本大小输出(请记住，此样本大小要求是针对每个变量的):

```
[n]230345
```

现在我们有了这些信息，我们可以确定实验需要进行多长时间。您需要输入的只是您的日平均流量:

```
average_daily_traffic <- 42000
```

运行 *days_calculator()* 函数:

```
days_calculator(sample_size_output, average_daily_traffic)
```

您将得到以下输出:

```
[1] It will take this many days to reach significance with your current traffic: 36
```

虽然此代码仅适用于 AB 设计的实验(即只有两个实验条件)，但是可以使用 *sample_size_calculator()* 中的 *pwr.anova.test()* 函数，替换 *pwr.2p.test()* ，修改给出的函数，以计算多个实验条件下所需的样本量。

## 结论

功耗分析是任何实验设计的必要方面。它允许分析师以给定的置信度确定检测给定规模的统计显著性效应所需的样本规模。相反，它也有助于在样本大小的限制下，以给定的置信度检测给定大小的效应。如果概率很低，建议改变实验的实验设计，或者将输入到功效分析中的某些数值最小化。

相互结合使用，计算所需的样本和实验持续时间可以为利益相关者提供非常有用的信息。获得这些信息可以帮助他们有效地规划他们的实验路线图。此外，这些预定的数字可以帮助确定某些实验的可行性，或者期望的提升是否过于理想化。