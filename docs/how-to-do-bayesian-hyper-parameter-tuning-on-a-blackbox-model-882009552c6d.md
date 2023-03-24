# 如何在黑盒模型上进行贝叶斯超参数调整

> 原文：<https://towardsdatascience.com/how-to-do-bayesian-hyper-parameter-tuning-on-a-blackbox-model-882009552c6d?source=collection_archive---------16----------------------->

## 云 ML 引擎上任意函数的优化

谷歌云 ML 引擎提供了一个使用贝叶斯方法的超参数调整服务。它不限于 TensorFlow 或 scikit-learn。事实上，它甚至不仅限于机器学习。您可以使用贝叶斯方法来调整几乎任何黑盒模型。

为了进行演示，我将调整一个流量模型，以找到能使给定流的延迟最小的流量配置。

![](img/fd89118f25ace8b11588e890593fc0f8.png)

I’ll demonstrate the hyper-parameter tuning on a traffic model. Photo by [Denys Nevozhai](https://unsplash.com/photos/7nrsVjvALnA?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/traffic?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

## 1.调用要优化的黑盒函数

为了简单起见，我将在 Python 本身中实现“黑盒函数”(它改编自几个[数值软件包](http://www.math.wpi.edu/saspdf/iml/chap11.pdf)中使用的一个例子)。在现实生活中，您可以调用任何可执行文件，因此您不局限于 Python。只要确保得到一个与您想要最小化或最大化的东西(在这个例子中是延迟)相对应的浮点值。

```
def compute_delay(flow, x12, x32):
    # traffic on other roads, assuming all traffic that arrives
    # at an intersection has to leave it
    x13 = flow - x12
    x24 = x12 + x32
    x34 = x13 - x32 # travel time on each road segment
    t12 = 5 + .1 * x12 / (1\. - x12 / 10.);
    t13 = x13 / (1\. - x13 / 30.);
    t32 = 1\. + x32 / (1\. - x32 / 10.);
    t24 = x24 / (1\. - x24 / 30.);
    t34 = 5 + .1 * x34 / (1\. - x34 / 10.);

    # total delay
    f = t12*x12 + t13*x13 + t32*x32 + t24*x24 + t34*x34;
    return(f);
```

给定节点间的交通流量和我们期望的车辆/小时总流量，上面的模型计算道路网络中的总交通延迟。

我们需要为某个流量值找到 x12 和 x32 的最佳值。

## 2.创建命令行参数

确保每个可优化的参数都是命令行参数:

```
 parser = argparse.ArgumentParser()
    parser.add_argument('--x12', type=float, required=True)
    **parser.add_argument('--x32', type=float, required=True)**
```

## 3.调用 hypertune 软件包

然后，写出我们希望使用 hypertune 软件包优化的指标:

```
 hpt = hypertune.HyperTune()
    **hpt.report_hyperparameter_tuning_metric**(
       **hyperparameter_metric_tag='delay'**,
       metric_value=delay,
       global_step=1)
```

## 4.编写 hyperparam.yaml 文件

编写一个配置文件。该文件应该包含要在其上运行代码的机器的类型、要优化的指标的名称以及要优化的每个参数的约束条件:

```
trainingInput:
  scaleTier: CUSTOM
  **masterType: standard  # machine-type**
  hyperparameters:
    goal: MINIMIZE
    maxTrials: 10
    maxParallelTrials: 2
    **hyperparameterMetricTag: delay**
    params:
    - parameterName: x12
      type: DOUBLE
      minValue: 0
      maxValue: 10
      scaleType: UNIT_LINEAR_SCALE
    **- parameterName: x32
      type: DOUBLE
      minValue: 0
      maxValue: 10
      scaleType: UNIT_LINEAR_SCALE**
```

## 5.将超参数调整作业提交给 ML 引擎

您可以使用 REST API 或 Python，但最简单的方法是从命令行使用 gcloud:

```
#!/bin/bash
BUCKET=<yourbuckethere>
JOBNAME=hparam_$(date -u +%y%m%d_%H%M%S)
REGION=us-central1
gcloud ml-engine jobs **submit training** $JOBNAME \
  --region=$REGION \
  --module-name=**trainer.flow** \
  --package-path=$(pwd)/trainer \
  --job-dir=gs://$BUCKET/hparam/ \
  **--config=hyperparam.yaml** \
  -- \
  --flow=5
```

## 6.在 GCP 控制台上查看结果

等待作业完成，并在 [GCP 控制台](https://console.cloud.google.com/mlengine)上查看结果。我有:

```
 {
      "trialId": "8",
      "hyperparameters": {
        **"x12": "0.70135653339854276",
        "x32": "0.018214831148084532"**
      },
      "finalMetric": {
        "trainingStep": "1",
        **"objectiveValue": 50.2831830645**
      }
    },
```

这种道路配置的实际最佳值是 40.3 的延迟；即使搜索空间很大并且问题是非线性的，我们在 10 次尝试中还是得到了相对接近的结果。

## 后续步骤:

1.  试试看。[的完整代码在 GitHub](https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/blogs/hparam) 上。
2.  阅读关于超参数调整的 ML 引擎[文档。](https://cloud.google.com/ml-engine/docs/tensorflow/hyperparameter-tuning-overview)
3.  阅读可用的[机器类型](https://cloud.google.com/ml-engine/docs/tensorflow/machine-types)的 ML 引擎文件。