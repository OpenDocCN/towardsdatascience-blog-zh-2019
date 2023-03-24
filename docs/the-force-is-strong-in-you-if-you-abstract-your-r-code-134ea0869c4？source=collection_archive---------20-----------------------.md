# 如果你抽象出你的 R 代码，这种力量在你身上会很强

> 原文：<https://towardsdatascience.com/the-force-is-strong-in-you-if-you-abstract-your-r-code-134ea0869c4?source=collection_archive---------20----------------------->

## 通过编写一个函数来分析星球大战中的角色，学习 R

任何体面的数学家或计算机程序员都会告诉你，如果一个任务被一次又一次地重复，那么它应该被做成一个函数。

这一直都是正确的，如果您仍然在一次又一次地编写重复的任务，而只是改变一两个变量——例如，如果您只是复制/粘贴代码——那么您需要立即停下来，学习如何编写函数。

但是最近的发展意味着有越来越多的动机去考虑你的代码的哪些部分可以被抽象。R 包的发展绕过了非标准的评估挑战，并通过 quosures 和相关的表达式增强了抽象能力，这意味着惊人的能力已经触手可及。

## R 中的函数

先简单说一下。如果您正在进行相同的分析，但只是更改变量值，则创建函数非常有用。让我们使用`dplyr`中的`starwars`数据集。如果我们想要一个所有人类角色的列表，我们可以使用这个:

```
starwars_humans <- starwars %>% 
  dplyr::filter(species == "Human") %>% 
  dplyr::select(name)
```

这将返回 35 个字符的名称。现在，如果我们想要一个相同的列表，但是有几个其他物种，我们可以复制粘贴并更改`species`的值。或者我们可以编写这个函数供将来使用:

```
species_search <- function(x) {
  starwars %>% 
    dplyr::filter(species == x) %>% 
    dplyr::select(name)
}
```

现在，如果我们运行`species_search("Droid")`，我们会得到一个四个字符的列表，并放心地看到我们的伙伴 R2-D2 在那里。

我们当然可以扩展它，使它成为一个具有多个变量的函数，以帮助我们基于各种条件进行搜索。

## 利用 rlang 的特性进一步抽象搜索

上面的问题是这个功能灵活性有限。它是以这样一种方式定义的，您无法控制要过滤哪个变量。

如果我们想重新定义这个函数，让它根据我们设置的任意条件返回一个列表呢？现在，我们可以为函数设置两个参数，一个表示要过滤的列，另一个表示要过滤的值。我们可以使用`rlang`中的`enquo`函数来捕获列名，以便在`dplyr::filter()`中使用。像这样:

```
starwars_search <- function(filter, value) {

  filter_val <- rlang::enquo(filter)

  starwars %>% 
    dplyr::filter_at(vars(!!filter_val), all_vars(. == value)) %>% 
    dplyr::select(name)
}
```

现在，如果我们评估`starwars_search(skin_color, "gold")`，我们会放心地看到我们焦虑但可爱的朋友 C-3PO 回来了。

## 甚至进一步允许使用 purrr 的任意过滤条件

因此，即使我们采取了上述步骤，我们已经使我们的搜索功能更加抽象和强大，但它仍然有些有限。例如，它只处理一个过滤器，并且只查找与该单个值匹配的字符。

假设我们有一组列表形式的过滤器。我们可以使用`purrr`中的`map2`函数获取该列表，并将其分解为一系列 quosure 表达式，这些表达式可以作为单独的语句传递给`dplyr::filter`，使用一个作用于数据帧的新函数:

```
my_filter <- function(df, filt_list){     
  cols = as.list(names(filt_list))
  conds = filt_list
  fp <- purrr::map2(cols, conds, 
                    function(x, y) rlang::quo((!!(as.name(x))) %in% !!y))
  dplyr::filter(df, !!!fp)
}
```

现在，这允许我们进一步抽象我们的`starwars_search`函数，以接收列表中的任意一组过滤条件，并且这些条件可以被设置为匹配向量中表示的一组值中的单个值:

```
starwars_search <- function(filter_list) {
  starwars %>% 
    my_filter(filter_list) %>% 
    dplyr::select(name)
}
```

例如，现在我们可以查找所有有蓝色或棕色眼睛、是人类、来自塔图因或奥德朗的角色，使用`starwars_search(list(eye_color = c("blue", “brown"), species = “Human", homeworld = c("Tatooine", “Alderaan")))`将返回以下内容:

```
# A tibble: 10 x 1
   name               
   <chr>              
 1 Luke Skywalker     
 2 Leia Organa        
 3 Owen Lars          
 4 Beru Whitesun lars 
 5 Biggs Darklighter  
 6 Anakin Skywalker   
 7 Shmi Skywalker     
 8 Cliegg Lars        
 9 Bail Prestor Organa
10 Raymus Antilles
```

现在你已经准备好释放原力的全部力量，通过开发抽象你的`dplyr`代码的多个元素的函数。例如，这里有一个函数可以让你找到你想要的某些星球大战角色的任何分组平均值:

```
starwars_average <- function(mean_col, grp, filter_list) { calc_var <- rlang::enquo(mean_col)
  grp_var <- rlang::enquo(grp)

  starwars %>% 
    my_filter(filter_list) %>% 
    dplyr::group_by(!!grp_var) %>% 
    summarise(mean = mean(!!calc_var, na.rm = TRUE))
}
```

因此，如果您想根据人类的家乡找到所有人类的平均身高，可以使用`starwars_average(height, homeworld, list(species = "Human"))`来完成，它将返回这个表:

```
# A tibble: 16 x 2
   homeworld     mean
   <chr>        <dbl>
 1 Alderaan      176.
 2 Bespin        175 
 3 Bestine IV    180 
 4 Chandrila     150 
 5 Concord Dawn  183 
 6 Corellia      175 
 7 Coruscant     168.
 8 Eriadu        180 
 9 Haruun Kal    188 
10 Kamino        183 
11 Naboo         168.
12 Serenno       193 
13 Socorro       177 
14 Stewjon       182 
15 Tatooine      179.
16 <NA>          193
```

虽然这是一个有点琐碎的例子，但我希望这能帮助您更好地理解当今 R 函数的潜力。当你审视自己的日常工作时，你会发现有机会将一些最常见的操作抽象成一些功能，从而节省你大量的时间和精力。实际上，我在这里展示的只是可能性的冰山一角。

最初我是一名纯粹的数学家，后来我成为了一名心理计量学家和数据科学家。我热衷于将所有这些学科的严谨性应用到复杂的人的问题上。我也是一个编码极客和日本 RPG 的超级粉丝。在[*LinkedIn*](https://www.linkedin.com/in/keith-mcnulty/)*或*[*Twitter*](https://twitter.com/dr_keithmcnulty)*上找我。*

*非常感谢我团队的 Sai Im，他用函数式编程的魔法启发了我们的一些想法。*

![](img/6ce4ad22d78408b22454038bd8f5de6d.png)