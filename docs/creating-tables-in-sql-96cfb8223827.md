# 在 SQL 中创建表

> 原文：<https://towardsdatascience.com/creating-tables-in-sql-96cfb8223827?source=collection_archive---------15----------------------->

理解 SQL 的语法非常简单。它遵循一个简单的结构，允许我们对不同的变量重用语法。以下是创建表格的一些基本语法。

# 正在创建数据库

```
CREATE DATABASE database_name;
```

# 创建表格

```
CREATE TABLE movies
(
     movie_name   VARCHAR(200),
     movie_year   INTEGER,
     country      VARCHAR(100), 
     genre        VARCHAR NOT NULL, 
     PRIMARY KEY  (movie_name, movie_year)
);
```

在括号内注明列名、列类型和长度。您可以声明 NOT NULL 以确保它总是被填充。

# 插入数据

```
INSERT INTO movies VALUES
('SE7EN', 1995, 'USA', 'Mystic, Crime'),
('INSTERSTELLAR', 2014, 'USA', 'Science Fiction'),
('The Green Mile', 1999, 'USA', 'Drama'),
('The Godfather', 1972, 'USA', 'CRIME')
```

如果输入的值类型不正确或者为空，并且不为 NULL，SQL 将抛出错误。

# 更新数据

如果您忘记添加一个值，您可以使用 update 更新它。

```
UPDATE movies
SET country = 'USA'
WHERE movie_name = 'The Godfather' AND movie_year = 1972;
```

# 更改表格

您可以更新项目并将其设置为默认值。

```
ALTER TABLE movies
ALTER COLUMN country SET DEFAULT 'USA';
```

测试默认选项。现在，无论何时你提到违约，你都会得到美国。

```
INSERT INTO movies VALUES
('test', 2010, DEFAULT, 'test')
```

# 添加列

```
ALTER TABLE movies
ADD COLUMN director VARCHAR(150)
```

# 更新表格

如果要为任何列添加新值，可以使用 UPDATE。

```
UPDATE movies
SET director = 'Christopher Nolan'
WHERE movie_name = 'Interstellar';
```

# 删除行

如果要删除一行，可以使用 WHERE 引用它。

```
DELETE FROM movies
WHERE movie_name = 'test'
```

# 删除列

```
ALTER TABLE movies
DROP director;
```

还有另一种方法你可以放下一个项目，它被称为下降级联。

DROP CASCADE →删除选定的项目以及依赖于它的所有内容。例如，如果我们在整个电影的数据库中使用 DROP CASCADE，所有内容都将被删除。