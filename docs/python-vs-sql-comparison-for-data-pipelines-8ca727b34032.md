# Python 和 SQL 构建数据管道的比较

> 原文：<https://towardsdatascience.com/python-vs-sql-comparison-for-data-pipelines-8ca727b34032?source=collection_archive---------3----------------------->

![](img/0129f726e45f74072f67bd6574039245.png)

作为一名 web 开发人员，我第一次接触数据库和 SQL 是使用对象关系模型(ORM)。我使用的是 Django 查询集 API，使用该接口的体验非常好。从那以后，我转变成了一名数据工程师，更多地参与到利用数据集来构建人工智能的工作中。我的职责是从用户应用程序中提取数据，并将其转化为可供数据科学家使用的东西，这个过程通常被称为 ETL。

正如故事所述，生产系统中的数据是混乱的，在任何人能够利用这些数据构建人工智能之前，需要进行大量的转换。有些 JSON 列每行有不同的模式，有些列包含混合的数据类型，有些行有错误的值。此外，还需要计算用户成为客户的时间以及他们在两次访问之间等待的时间。当我着手清理、聚合和设计数据特性时，我试图决定哪种语言最适合这项任务。在我的工作中，我每天都要整天使用 python，我知道它可以完成这项工作。然而，我从这次经历中学到的是，仅仅因为 python 可以完成这项工作并不意味着它应该这样做。

# 我第一次误判 SQL 是在我认为 SQL 不能完成复杂转换的时候

我们正在使用一个时间序列数据集，我们希望在一段时间内跟踪特定的用户。隐私法阻止我们知道用户访问的具体日期，所以我们决定将记录的日期标准化为用户第一次访问的日期(即第一次访问后 5 天等)。).对于我们的分析，了解自上次访问以来的时间以及自首次访问以来的时间是很重要的。a 有两个样本数据集，一个有大约 750 万行，大小为 6.5 GBs，另一个有 550 000 行，大小为 900 MB。

使用下面的 python 和 SQL 代码，我使用较小的数据集首先测试转换。Python 和 SQL 分别用 591 和 40.9 秒完成了任务。这意味着 SQL 能够提供大约 14.5 倍的速度提升！

```
# PYTHON
# connect to db using wrapper around psycopg2
db = DatabaseConnection(db='db', user='username', password='password')# grab data from db and load into memory
df = db.run_query("SELECT * FROM cleaned_table;")
df = pd.DataFrame(df, columns=['user_id', 'series_id', 'timestamp'])# calculate time since first visit
df = df.assign(time_since_first=df.groupby('user_id', sort=False).timestamp.apply(lambda x: x - x.min()))# calculate time since last visit
df = df.assign(time_since_last=df.sort_values(['timestamp'], ascending=True).groupby('user_id', sort=False)['timestamp'].transform(pd.Series.diff))# save df to compressed csv
df.to_csv('transform_time_test.gz', compression='gzip') -- SQL equivalent
-- increase the working memory (be careful with this)
set work_mem='600MB';-- create a dual index on the partition
CREATE INDEX IF NOT EXISTS user_time_index ON table(user_id, timestamp);-- calculate time since last visit and time since first visit in one pass 
SELECT *, AGE(timestamp, LAG(timestamp, 1, timestamp) OVER w) AS time_since_last, AGE(timestamp, FIRST_VALUE(timestamp) OVER w) AS time_since_first FROM table WINDOW w AS (PARTITION BY user_id ORDER BY timestamp);
```

这种 SQL 转换不仅速度更快，而且代码可读性更好，因此更易于维护。这里，我使用了 lag 和 first_value 函数来查找用户历史中的特定记录(称为分区)。然后，我使用年龄函数来确定访问之间的时间差。

更有趣的是，当这些转换脚本应用于 6.5 GB 数据集时，python 完全失败了。在 3 次尝试中，python 崩溃了 2 次，我的电脑第三次完全死机了…而 SQL 用了 226 秒。

更多信息:
[https://www.postgresql.org/docs/9.5/functions-window.html](https://www.postgresql.org/docs/9.5/functions-window.html)
[http://www . PostgreSQL tutorial . com/PostgreSQL-window-function/](http://www.postgresqltutorial.com/postgresql-window-function/)

# 我第二次误判 SQL 是在我认为它不能展平不规则 json 的时候

对我来说，另一个改变游戏规则的因素是意识到 Postgres 与 JSON 配合得非常好。我最初认为在 postgres 中不可能展平或解析 json 我不敢相信我竟然这么傻。如果您想要关联 json，并且它的模式在行之间是一致的，那么您最好的选择可能是使用 Postgres 内置的能力来解析 json。

```
-- SQL (the -> syntax is how you parse json)
SELECT user_json->'info'->>'name' as user_name FROM user_table;
```

另一方面，我的样本数据集中有一半的 json 不是有效的 json，因此被存储为文本。在这种情况下，我面临一个选择，要么重新编码数据使其有效，要么删除不符合规则的行。为此，我创建了一个名为 is_json 的新 SQL 函数，然后可以用它在 WHERE 子句中限定有效的 json。

```
-- SQL
create or replace function is_json(text)
returns boolean language plpgsql immutable as $$
begin
    perform $1::json;
    return true;
exception
    when invalid_text_representation then 
        return false;
end $$;SELECT user_json->'info'->>'name' as user_name FROM user_table WHERE is_json(user_json);
```

不幸的是，我发现 user_json 有不同的模式，这取决于用户使用的应用程序版本。尽管从应用程序开发的角度来看，这是有意义的，但是有条件地解析每行的每种可能性的成本确实很高。我注定要再次进入 python 吗…一点机会都没有！我在 stack-overflow 上发现了另一个函数，是一个叫 [klin](https://stackoverflow.com/users/1995738/klin) 的 postgres 大神写的。

```
-- SQL
create or replace function create_jsonb_flat_view
    (table_name text, regular_columns text, json_column text)
    returns text language plpgsql as $$
declare
    cols text;
begin
    execute format ($ex$
        select string_agg(format('%2$s->>%%1$L "%%1$s"', key), ', ')
        from (
            select distinct key
            from %1$s, jsonb_each(%2$s)
            order by 1
            ) s;
        $ex$, table_name, json_column)
    into cols;
    execute format($ex$
        drop view if exists %1$s_view;
        create view %1$s_view as 
        select %2$s, %3$s from %1$s
        $ex$, table_name, regular_columns, cols);
    return cols;
end $$;
```

这个函数能够成功地展平我的 json，并相当容易地解决我最糟糕的噩梦。

最终意见

有一个习语宣称 Python 是几乎做任何事情的第二好语言。我相信这是真的，在某些情况下，我发现 Python 和“最佳”语言之间的性能差异可以忽略不计。然而，在这种情况下，python 无法与 SQL 竞争。这些认识以及我所做的阅读完全改变了我的 ETL 方法。我现在的工作模式是“不要把数据移动到代码，把代码移动到你的数据”。Python 将数据移动到代码中，而 SQL 就地对其进行操作。更重要的是，我知道我只是触及了 sql 和 postgres 能力的皮毛。我期待着更多令人敬畏的功能，以及通过使用分析仓库来提高速度的可能性。