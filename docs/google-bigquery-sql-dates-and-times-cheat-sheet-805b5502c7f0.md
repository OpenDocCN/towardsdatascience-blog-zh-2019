# Google BigQuery SQL 日期和时间备忘单

> 原文：<https://towardsdatascience.com/google-bigquery-sql-dates-and-times-cheat-sheet-805b5502c7f0?source=collection_archive---------4----------------------->

## 常见 BigQuery 日期和时间表达式的备忘单

![](img/11d421a458e09229d009090022ca0c21.png)

Photo by [Sonja Langford](https://unsplash.com/photos/eIkbSc3SDtI?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/clock?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

我今天早上刚刚开始做一个时间序列预测项目。与大多数数据科学任务一样，这个项目的第一步是首先收集数据。这意味着 Google BigQuery 中的 SQL 查询相对简单。我又一次发现自己在谷歌上搜索如何从时间戳中提取年份，并在文档中滚动以找到正确的函数，然后意识到我需要在某个地方写下来。我把它写在博客上，这样我就有东西可以参考，希望这也能帮助其他人。请注意，所有表达式都基于标准 SQL。

# 日期

## 日期部分

下列表达式中可以使用的所有日期部分的列表:

```
DAYOFWEEK (returns 1-7 Sunday is 1)
DAY
DAYOFYEAR (0-365)
WEEK (week of year 0-53, week begins on Sunday)
WEEK(<chosen weekday>) (week of year begins on your chosen day e.g. SUNDAY)
ISOWEEK (ISO 8601 week number, week begins on Monday)
MONTH
QUARTER (1-4)
YEAR (ISO 8601 year number) 
```

## 提取日期部分

```
EXTRACT(part FROM date_expression)Example: EXTRACT(YEAR FROM 2019-04-01)Output: 2019
```

## 从整数构造一个日期

```
DATE(year, month, day)Example: DATE(2019, 04, 01)Output: 2019-04-01
```

## 从日期中加减

```
DATE_ADD(date_expression, INTERVAL INT64_expr date_part)Example: DATE_ADD('2019-04-01', INTERVAL 1 DAY)Output: 2019-04-02DATE_SUB(date_expression, INTERVAL INT64_expr date_part)Example: DATE_SUB('2019-04-01', INTERVAL 1 DAY)Output: 2019-03-31Example use case - dynamic dates:where my_date between DATE_SUB(current_date, INTERVAL 7 DAY) and DATE_SUB(current_date, INTERVAL 1 DAY)
```

## 两个日期之间的差异

```
DATE_DIFF(date_expression, date_expression, date_part)Example: DATE_DIFF(2019-02-02, 2019-02-01, DAY)Output: 1
```

## 指定日期的粒度

```
DATE_TRUNC(date_expression, date_part)Example: DATE_TRUNC(2019-04-12, WEEK)Output: 2019-04-07
```

# 英国泰晤士报(1785 年创刊)

## 时间部分

```
MICROSECOND
MILLISECOND
SECOND
MINUTE
HOUR
```

## 从整数构造一个日期时间对象

```
DATETIME(year, month, day, hour, minute, second)
DATETIME(date_expression, time_expression)
DATETIME(timestamp_expression [, timezone])Example: DATETIME(2019, 04, 01, 11, 55, 00)Output: 2019-04-01 11:55:00
```

## 加减时间

```
DATETIME_ADD(datetime_expression, INTERVAL INT64_expr part)Example: DATETIME_ADD('2019-04-01 11:55:00', INTERVAL 1 MINUTE)Output: 2019-04-01 11:56:00DATETIME_SUB(datetime_expression, INTERVAL INT64_expr part)Example: DATETIME_SUB('2019-04-01 11:55:00', INTERVAL 1 MINUTE)Output: 2019-04-01 11:54:00
```

## 两次之间的差异

```
DATETIME_DIFF(datetime_expression, datetime_expression, part)Example: DATETIME_DIFF('2019-04-01 11:56:00', '2019-04-01 11:55:00', MINUTE)Output: 1
```

## 指定时间的粒度

```
DATETIME_TRUNC(datetime_expression, part)Example: DATETIME_TRUNC('2019-04-01 11:55:00', HOUR)Output: 2019-04-01 11:00:00
```

这绝不是 BigQuery 中日期和时间表达式的详尽指南。它只是作为我最常用的快速参考。如需更全面的指南，请参阅谷歌大查询[文档](https://cloud.google.com/bigquery/docs/reference/standard-sql/datetime_functions#parse_datetime)。