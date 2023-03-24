# 如何使用 R 正确解释连续变量和分类变量的相互作用

> 原文：<https://towardsdatascience.com/how-to-correctly-interpret-your-continuous-and-categorical-variable-interactions-in-regressions-51e5eed5de1e?source=collection_archive---------4----------------------->

## 理解你的回归

## 当你有互动时，解释系数可能是棘手的。你对它们的解释正确吗？

![](img/e501d3e2a812d46c0b3816b3a86eb6ad.png)

Photo by [Jon Robinson](https://unsplash.com/@jnr1963?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

理解和解释回归结果的能力是有效数据分析的基础。了解每个术语在模型规范中是如何表示的，对于准确解释模型的结果至关重要。然而，通常很容易通过查看 t 统计或 p 值得出结论，并假设模型做了您想要它做的事情，而没有真正理解在引擎盖下发生了什么。我在给本科生讲授统计学时经常观察到的一个错误是，当包含带有分类变量的交互项时，如何解释连续变量的主要效应。

这里我提供了一些 R 代码来演示为什么你不能简单地把系数解释为主要效果，除非你指定了一个对比度。

## TLDR:当你指定你的分类变量是一个以 0 为中心的对比变量时，你应该只把与分类变量相互作用的连续变量的系数解释为平均主效应。如果分类变量是虚拟编码的，你不能把它解释为平均主效应。

# 步骤 1:模拟数据

为了举例说明，我将创建一个包含变量`Income`、`Age`和`Gender`的假数据集。我的规范是，对于男性，`Income`和`Age`的相关系数 r = .80，对于女性，`Income`和`Age`的相关系数 r = .30。

从这个规格来看，**`**Age**`**对** `**Income**` **的平均作用，对** `**Gender**` **的控制应为. 55 (= (.80 + .30) / 2)。**至于平均群体差异，假设男性平均收入 2 美元，女性平均收入 3 美元。**

**分别生成每个性别的数据，然后连接起来创建一个组合数据帧:`data`。使用包`MASS`中的`mvrnorm`在`R`中生成数据:**

**这段代码还检查随机生成的数据是否具有我们指定的相关性和平均值。**

```
## Correlation between Income & Age for Male: 0.8 
## Correlation between Income & Age for Female: 0.3 
## Mean Income for Male: 2 
## Mean Income for Female: 3
```

# **第二步:错误地估计你的主要效果**

**现在我们有了样本数据，让我们看看当我们天真地运行一个线性模型，从`Age`、`Gender`以及它们的相互作用预测`Income`时会发生什么。**

```
## 
## Call: 
## lm(formula = "Income~Age*Gender", data = data) 
## 
## Residuals: 
##     Min      1Q  Median      3Q     Max 
## -3.4916 -0.4905 -0.0051  0.5044  3.2038 
## 
## Coefficients: 
##                Estimate Std. Error t value Pr(>|t|)     
## (Intercept)     3.00000    0.02521  118.99   <2e-16 *** 
## Age             0.30000    0.02522   11.89   <2e-16 *** 
## GenderMale     -1.00000    0.03565  -28.05   <2e-16 *** 
## Age:GenderMale  0.50000    0.03567   14.02   <2e-16 *** 
## --- 
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1 
## ## Residual standard error: 0.7973 on 1996 degrees of freedom 
## Multiple R-squared:  0.4921, Adjusted R-squared:  0.4913 
## F-statistic: 644.6 on 3 and 1996 DF,  p-value: < 2.2e-16
```

**上面的模型摘要打印了`Intercept`、`Age`、`GenderMale`、`Age:GenderMale`的系数。我们可以清楚地看到,`Age`的效果是 0 . 30，这当然不是控制性别的平均效果，而仅仅是对女性群体的效果。`GenderMale`的效果是$-1，即男性组比女性组少挣多少钱，截距为$3。最后，相互作用`Age:GenderMale`代表*与`Age`的相关性男性比女性多多少*(0.5 = 0.8-0.3)。**

**重要的是，这是带有分类变量的默认 R 行为，它**按字母顺序将第一个变量设置为参考级别*(即截距)。所以在我们的例子中，女性被设定为我们的参考水平。然后，我们的分类变量被虚拟编码(也称为治疗对比)，因此女性为 0，男性为 1，这可以通过使用函数`contrasts`来验证。**

```
contrasts(data**$**Gender)##        Male 
## Female    0 
## Male      1
```

# **第三步:估计你的主要效果的正确方法**

**那么，我们需要做什么来获得`Age`对`Income`控制`Gender`的*平均*效果，同时保持交互？答案是:**指定以 0 为中心的对比度，使女性编码为-.50，男性编码为. 50。让我们看看当我们指定对比度并重新运行我们的模型时会发生什么。****

```
## 
## Call: 
## lm(formula = "Income~Age*Gender", data = data) 
## 
## Residuals: 
##     Min      1Q  Median      3Q     Max 
## -3.4916 -0.4905 -0.0051  0.5044  3.2038 
## 
## Coefficients: 
##             Estimate Std. Error t value Pr(>|t|)     
## (Intercept)  2.50000    0.01783  140.23   <2e-16 *** 
## Age          0.55000    0.01784   30.84   <2e-16 *** 
## Gender1     -1.00000    0.03565  -28.05   <2e-16 *** 
## Age:Gender1  0.50000    0.03567   14.02   <2e-16 *** 
## --- 
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1 
## 
## Residual standard error: 0.7973 on 1996 degrees of freedom 
## Multiple R-squared:  0.4921, Adjusted R-squared:  0.4913 
## F-statistic: 644.6 on 3 and 1996 DF,  p-value: < 2.2e-16
```

**我们再次得到四个术语，但是它们被指定为`Intercept`、`Age`、`Gender1`和`Age:Gender1`。`Age`效应是 **0.55，这正是我们在生成数据时指定的跨性别平均效应**(0.55 =(0.8+0.3)/2)。`Age:Gender1`交互作用为 0.5，这是性别间年龄效应的**差异**(0.5 = 0.8–0.3)。`Gender1`的效果是$-1，它代表两性之间的平均差异($2-$3)，正如我们的对比所指定的。参考值`Intercept`是 2.5 美元，这是不同性别的平均收入(($2+$3) / 2)。**

**再一次，我们可以用下面的例子来验证我们的对比:**

```
contrasts(data**$**Gender)##        [,1]
## Female -0.5
## Male    0.5
```

# **结论**

**我希望这个例子表明，当你建立连续变量和分类变量之间相互作用的线性模型时，你需要注意它们是如何指定的(虚拟编码或对比)，因为这将改变你解释系数的方式。**

**最重要的是，当您已经将分类变量指定为以 0 为中心的对比度时，您应该只将与分类变量相互作用的连续变量的系数解释为平均主效应。如果分类变量是虚拟编码的，你不能把它解释为主要效应，因为它们成为参考水平效应的估计值。**

**如需了解更多信息，请访问 [stackexchange](https://stats.stackexchange.com/questions/41129/interpreting-coefficients-of-an-interaction-between-categorical-and-continuous-v) 和 [r-bloggers](https://www.r-bloggers.com/interpreting-interaction-coefficient-in-r-part1-lm/) 查看这个问题的更多答案，这个问题已经在网上被多次询问。**

## **附加信息**

**如果希望分类变量被视为虚拟代码，可以将其设置为处理对比。它复制了 r 提供的默认结果。**

```
## 
## Call: 
## lm(formula = "Income~Age*Gender", data = data) 
## 
## Residuals: 
##     Min      1Q  Median      3Q     Max 
## -3.4916 -0.4905 -0.0051  0.5044  3.2038 
## 
## Coefficients: 
##             Estimate Std. Error t value Pr(>|t|)     
## (Intercept)  3.00000    0.02521  118.99   <2e-16 *** 
## Age          0.30000    0.02522   11.89   <2e-16 *** 
## Gender2     -1.00000    0.03565  -28.05   <2e-16 *** 
## Age:Gender2  0.50000    0.03567   14.02   <2e-16 *** 
## --- 
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1 ## 
## Residual standard error: 0.7973 on 1996 degrees of freedom 
## Multiple R-squared:  0.4921, Adjusted R-squared:  0.4913 
## F-statistic: 644.6 on 3 and 1996 DF,  p-value: < 2.2e-16
```

**如果你运行模型*而没有*交互作用，那么即使你的分类变量是虚拟编码的，年龄的主要影响也是如你所料控制性别的**平均影响。****

```
## 
## Call: 
## lm(formula = "Income~Age+Gender", data = data) 
## 
## Residuals: 
##     Min      1Q  Median      3Q     Max 
## -3.5525 -0.5102 -0.0055  0.5424  3.2461 
## 
## Coefficients: 
##             Estimate Std. Error t value Pr(>|t|)     
## (Intercept)  3.00000    0.02642  113.56   <2e-16 *** 
## Age          0.55000    0.01869   29.43   <2e-16 *** 
## Gender2     -1.00000    0.03736  -26.77   <2e-16 *** 
## --- 
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1 
## 
## Residual standard error: 0.8354 on 1997 degrees of freedom 
## Multiple R-squared:  0.4421, Adjusted R-squared:  0.4416 
## F-statistic: 791.3 on 2 and 1997 DF,  p-value: < 2.2e-16
```

**感谢您的阅读，并随时查看我的其他与数据科学相关的帖子。**

**[](/why-models-with-significant-variables-can-be-useless-predictors-3354722a4c05) [## 为什么有重要变量的模型可能是无用的预测器

### 统计模型中的重要变量不能保证预测性能

towardsdatascience.com](/why-models-with-significant-variables-can-be-useless-predictors-3354722a4c05) [](/chance-is-not-enough-evaluating-model-significance-with-permutations-e3b17de6ba04) [## 机会是不够的:用排列评估模型的重要性

### 当训练机器学习模型进行分类时，研究人员和数据科学家经常比较他们的模型…

towardsdatascience.com](/chance-is-not-enough-evaluating-model-significance-with-permutations-e3b17de6ba04)**