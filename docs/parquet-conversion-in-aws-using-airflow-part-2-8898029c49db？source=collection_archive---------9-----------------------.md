# 利用气流在 AWS 中转换拼花地板(第二部分)

> 原文：<https://towardsdatascience.com/parquet-conversion-in-aws-using-airflow-part-2-8898029c49db?source=collection_archive---------9----------------------->

在本帖中，我们将深入探讨定制的气流操作符，并了解如何轻松处理气流中的拼花转换。

如果您使用 AWS，主要有三种方法可以将红移/S3 中的数据转换为拼花文件格式:

*   与下一个选项相比，使用 Pyarrow 可能会花费一些时间，但在分析数据时会有更大的自由度，且不涉及额外的成本。
*   使用雅典娜 CTAS，这涉及扫描数据的成本，但全面更快地转换数据。
*   使用 Unload 来直接拼花 AWS 尚未发布的拼花。

![](img/68acdebaa4bbd2ec5bc76d34f0617c61.png)

大约有 5 种这样的方法进行了概念验证，但是上面的前两种方法在速度、效率和成本方面非常突出。我们将探索第一个选项，我们将使用代码片段来了解哪些是转换为拼花地板的基本部分。以下是转换它所涉及的步骤:

1.  **计算要创建的分区:**从 SVV _ 外部 _ 分区表中提取分区值，并计算需要创建哪些分区。
2.  **卸载到 S3:** 现在，要使用 Pyarrow，我们需要 S3 中的数据。因此，第二步是在 Redshift 中使用“Unload”命令，如果在 s3 中还没有为上一步中所有需要的分区加载数据。我们可以使用任何适合数据的格式，如“TSV”。
3.  **解析成所需的数据类型:**之后，我们需要将 s3 数据转换成 parquet 兼容的数据类型，这样在通过外部表使用它时就不会出现任何错误。这可以通过在 dataframe 中转换 pandas 数据类型或 parquet 数据类型来实现。

为了简单起见，我们将使用红移谱将分区加载到它的外部表中，但是下面的步骤可以用于 Athena 外部表的情况。

# **计算要创建的分区**

借助于 SVV _ 外部 _ 分区表，我们可以计算出哪些分区已经全部存在以及哪些分区都需要被执行。下面的代码片段使用 CustomRedshiftOperator，它实际上使用 PostgresHook 来执行 Redshift 中的查询。xcom_push 将查询结果推入 xcom，我们可以读取并检查“dummy_external_table”中的分区。

```
check_partition_already_exists_task = CustomRedshiftOperator(
    task_id='check_batch_already_exists',
    redshift_conn_id='redshift_default',
    sql='select * from SVV_EXTERNAL_PARTITIONS where tablename = \'dummy_external_table\' and values like \'%2019-01-01%\',
    xcom_push=True,
    retries=3
)
```

# 卸载到 s3

根据需要计算的分区，我们可以形成可以在 Unload 语句中使用的查询。下面的代码片段在自定义操作符中使用 PostgresHook，该操作符采用 select_query、s3_unload_path、required IAM role 和 unload_options 将 Redshift 中的数据填充到 s3 中(如果该数据还不存在)。

select_query =这可以是' select * from schema . dummy _ table where sort _ key between ' from _ time ' and ' to _ time ' '

s3_unload_path =这将是 s3 中 unload 推送文件块的路径，我们需要进一步使用这些文件块进行转换。例如:S3://bucket/source _ name/schema _ name/table _ name/landing/

unload_options =转义头分隔符' \ \ t ' allow overwrite max filesize AS 275。我更喜欢在选项中给出 maxfilesize，因为我不想使用 Airflow 实例的所有 RAM，否则 AWs 将使用默认文件大小 6GB。

```
unload_query = """
            UNLOAD ( $$ {select_query} $$)
            TO '{s3_unload_path}'
            iam_role '{iam_role}'
            {unload_options};
            """.format(select_query=select_query,
                       s3_unload_path=self.s3_unload_path,
                       iam_role=self.iam_role,
                       unload_options=self.unload_options)
self.hook.run(unload_query, self.autocommit)
self.log.info("UNLOAD command complete...")
```

作为最佳实践，让我们遵循 s3 结构，其中带有“landing”的前缀将充当卸载查询加载红移数据的着陆区域，而“processed”将用作放置转换后的拼花数据的键。

# 解析成所需的数据类型

下面的代码片段将熊猫数据帧转换成 pyarrow 数据集，然后将其加载到 s3 中。

```
table = pa.Table.from_pandas(df)
buf = pa.BufferOutputStream()
pq.write_table(table, buf, compression='snappy', coerce_timestamps='ms', allow_truncated_timestamps=True)self.s3.load_bytes(bytes_data=buf.getvalue().to_pybytes(), key=self.s3_key + '_' + str(suffix) + '.snappy.parquet', bucket_name=self.s3_bucket, replace=True)
```

到目前为止一切顺利。之后，需要将分区添加到外部表中，瞧！选择查询将显示其中的数据。“添加分区”还将使用*customredshiftopoperator*，它将在红移集群上运行添加分区查询。

现在，当熊猫数据帧中的特定列具有混合数据类型或者该列中有“NaNs”时，问题就出现了。在这种情况下，pandas 将把列数据类型作为对象“O ”,当 pyarrow 用于 pandas 数据帧时，pyarrow 将使用 pandas 提供的列数据类型，并将其转换为自己的数据类型。在对象数据类型的情况下，它会将其转换为自己的“二进制”，这本质上不是该列的原始数据类型。因此，当您在外部表上执行 select 查询时，它会向您显示解析错误。

因此，为了避免这种错误，我们需要将 pandas dataframe 列“转换”为所需的数据类型，或者在 parquet 中使用 parse_schema 将其隐式转换为 parquet 格式。在这篇文章中，我们将探索转换数据类型的“astype”方法。

在执行代码片段之前，应该有“datetime_col_list ”,它包含日期时间格式的所有列名，等等，用于 bool_col_list 和 integer_col_list。Integer_col_list 中也有浮动列。datetime_col_dict_list 是一个 dict 列表，类似于[{"app_id": "datetime64[ms]"}]，类似于 bool 和 integer。

下面的代码片段将首先直接输入列，如果有任何错误，它将逐列输入，如果仍然有任何错误，则逐行输入。如果仍然存在任何错误，则删除该记录。就性能而言，它在 12 分钟内卸载+转换 14GB 的数据，我认为这是可以管理的。此外，我们可以将下面的代码片段放在一个函数中，并针对日期时间和整数调用它。

```
try:
   df[datetime_col_list] = df[datetime_col_list].astype('datetime64[ms]')
except ValueError:
    for schema_dict in datetime_col_dict_list:
        try:
            should_restart = True
            while should_restart:
                should_restart = False
                try:
                    df = df.astype(schema_dict)
                except ValueError as e:
                    self.log.warn('ValueError - reason: ' + str(e))
                    for i, item in enumerate(df[list(schema_dict.keys())[0]]):
                        try:
                            if not pd.isnull(item):
                                pd.to_datetime(item)
                        except ValueError:
                            logger.info('Corrupted row at index {}: {!r}'.format(i, item))
                            logger.info(df.loc[i, 'event_id'])
                            df.drop([i], inplace=True)
                            should_restart = True
        except KeyError as e:
            logger.info(schema_dict)
            logger.warn('KeyError - reason: ' + str(e))
            continuefor col_name in bool_col_list:
    df[col_name] = df[col_name].map({'f': 'false', 't': 'true'})
    df[col_name] = df[col_name].fillna('').astype('str')for schema_dict in integer_col_dict_list:
    try:
        should_restart = True
        while should_restart:
            should_restart = False
            try:
                df = df.astype(schema_dict)
            except ValueError as e:
                logger.warn('ValueError - reason: ' + str(e))
                for i, item in enumerate(df[list(schema_dict.keys())[0]]):
                    try:
                        if not pd.isnull(item):
                            float(item)
                    except ValueError:
                        logger.info('Corrupted row at index {}: {!r} for column: {col}'.format(i, item, col=list(schema_dict.keys())[0]))
                        logger.info(df.loc[i, 'event_id'])
                        df.drop([i], inplace=True)
                        should_restart = True
     except KeyError as e:
         logger.info(schema_dict)
         logger.warn('KeyError - reason: ' + str(e))
         continue
```

# 定制气流操作员

最后，上面的 3 个片段被包装在自定义操作符中，我们只需要提供必要的细节，它就会自动计算所需的分区，在 s3 中为每个分区创建 parquet 文件，并将分区添加到外部表中。

```
to_parquet_task = CustomRedshiftToS3Transfer(
    task_id='to_parquet',
    redshift_conn_id='redshift_default',
    schema='___SCHEMA___',
    table='___TABLE_NAME___',
    s3_key=parquet_key,
    where_clause="{col} >= \'{from_time}\' and {col} < \'{to_time}\'".format(
        col=redshift_sort_key,
        # col=batch_id_col,
        from_time=str(pd.to_datetime(to_check, format='%Y-%m-%d')),
        to_time=str(pd.to_datetime(to_check, format='%Y-%m-%d') + timedelta(days=1))
    ),
    s3_bucket=Variable.get('bucket'),
    parse_schema=table_schema,
    unload_options=parquet_unload_options,
    aws_conn_id='s3_etl',
    is_parquet=True,
    engine='pyarrow',
    retries=3
)
```

我在这里使用了‘is _ parquet ’,因为上面的自定义操作符也处理其他数据格式。“where_clause”将构成 unload 语句中 select 查询的一部分，并将使用 from_time 和 to_time 从 redshift 表中选择所需的数据片。

# 包裹

因此，前一篇文章和这篇文章提供了一些关于什么是 parquet 文件格式，如何在 s3 中组织数据，以及如何使用 Pyarrow 有效地创建 parquet 分区的想法。上面的自定义操作符也有“引擎”选项，可以指定是使用“pyarrow”还是使用“athena”将记录转换为拼花。雅典娜选项将自动选择雅典娜 CTAS 选项，将 s3 中的卸载数据转换为 s3 中的拼花数据。

这里是第一部分:[https://towardsdatascience . com/parquet-conversion-in-AWS-using-air flow-part-1-66 ADC 0485405](/parquet-conversion-in-aws-using-airflow-part-1-66adc0485405)