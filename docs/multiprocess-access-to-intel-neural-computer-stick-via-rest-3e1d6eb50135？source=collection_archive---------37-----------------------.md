# 通过 REST 对英特尔神经计算机棒进行多进程访问

> 原文：<https://towardsdatascience.com/multiprocess-access-to-intel-neural-computer-stick-via-rest-3e1d6eb50135?source=collection_archive---------37----------------------->

## 用 Python 包装器扩展硬件限制

# 单一过程问题

我开始在我的树莓派机器人上使用[英特尔 NCS，这种升级有积极和消极的一面。](/robot-tank-with-raspberry-pi-and-intel-neural-computer-stick-2-77263ca7a1c7)

积极的一面是 NCS 能够用 Tensorflow 和 OpenCV 替换所有运行在 Raspberry 上的网络。

![](img/ea0fb87340c6b81a18d8bfb1a37c727f.png)

性能的提高激发了新的目标，但是很快我发现 NCS 不能在两个不同的过程中使用。

```
E: [ncAPI] [    926029] resetAll:348     Failed to connect to stalled device, rc: X_LINK_ERROR 
E: [ncAPI] [    933282] ncDeviceOpen:672        Failed to find suitable device, rc: X_LINK_DEVICE_NOT_FOUND
```

在英特尔支持论坛上的搜索带来了一个[类似的问题](https://software.intel.com/en-us/forums/computer-vision/topic/815657)。

有一个关于[供应商文档](https://docs.openvinotoolkit.org/2019_R1.1/_docs_IE_DG_supported_plugins_MYRIAD.html#supported_configuration_parameters)的参考，它说得很简单:

> 单个设备不能跨多个进程共享。

这意味着有必要建立一个变通办法。

# NCS 服务

所以这个想法是将 NCS 的工作委托给一个专门的服务。

该服务应该提供 REST API 包装 NCS 函数。

## NCS API

基本用例非常简单:

*   加载模型
*   进行推理
*   列出加载的模型
*   获取模型的特征

最有趣的场景是推理运行。

通常它返回整个模型的输出——多维张量。

原始张量在某些情况下是有用的，但通常我们需要更具体的数据。

我使用 NCS 对图像进行分类，检测物体和分割道路。

因此，一般用例“运行推理”被扩展为:

*   分类
*   发现
*   段

## REST 接口

*   POST: /load —加载模型
*   POST: /unload/$model —删除模型(从服务内存中，无法从设备中删除)
*   GET: /list —列出模型
*   GET:/input/shape/$ model-获取模型的形状
*   POST: /inference/file/$model —使用来自内存的数据运行推理
*   POST: /inference/path/$model —使用文件系统中的数据运行推理
*   POST: /classify/file/$model
*   POST: /classify/path/$model
*   POST: /detect/file/$model
*   POST: /detect/path/$model
*   POST: /segment/file/$model
*   POST: /segment/path/$model

## 内存与文件系统

有两种方法可以传递图像——通过内存(如果已经有了，可能会有用)或文件系统路径。

当 NCS 服务和客户端运行在同一个 Raspberry Pi 上时，第二种方法会更好。

有一个简短的基准确认(1000 次尝试):

内存:87.5 秒
文件:63.3150 秒

## 分类

方法/分类将原始推理输出转换成一组对(类，分数):

```
def get_class_tensor(data):
    ret = []
    thr = 0.01
    while(True):
        cls = np.argmax(data)
        if data[cls] < thr:
            break;
        logging.debug(("Class", cls, "score", data[cls]))
        c = {"class" : int(cls), "score" : int(100 * data[cls])}
        data[cls] = 0
        ret.append(c)
    return retdef classify(model_id, img):
    rc, out = run_inference(model_id, img)
    if not rc:
        return rc, out
    return True, get_class_tensor(out)
```

## 侦查

输出检测张量包含一组(类别、概率、归一化坐标),看起来不太可读。

将其转换为简单的表示，同时删除最不可能的选项:

```
def get_detect_from_tensor(t, rows, cols):
    score = int(100 * t[2])
    cls = int(t[1])
    left = int(t[3] * cols)
    top = int(t[4] * rows)
    right = int(t[5] * cols)
    bottom = int(t[6] * rows)
    return {"class" : cls, "score" : score, "x" : left, "y" : top, "w" : (right - left), "h" : (bottom - top)} def build_detection(data, thr, rows, cols):
    T = {}
    for t in data:
        score = t[2]
        if score > thr:
            cls = int(t[1])
            if cls not in T:
                T[cls] = get_detect_from_tensor(t, rows, cols)
            else:
                a = T[cls]
                if a["score"] < score:
                    T[cls] = get_detect_from_tensor(t, rows, cols)
    return list(T.values())
```

## 分割

分段张量包含模型维度内分类的概率。

将其转换为类别掩码:

```
def segment(model_id, img):
    rc, out = run_inference(model_id, img)
    if not rc:
        return rc, out
    out = np.argmax(out, axis=0)
    out = cv.resize(out,(img.shape[1], img.shape[0]), interpolation=cv.INTER_NEAREST)
    return True, out
```

## 结论

我根据自己的需要开发了在 Raspberry Pi 上运行的服务。

但是没有什么可以阻止在任何其他平台上运行 Python、OpenVino 和 NCS。

## 链接

[英特尔 OpenVino 简介](https://software.intel.com/en-us/articles/run-intel-openvino-models-on-intel-neural-compute-stick-2)

【Raspbian 官方 OpenVino 安装指南

[奥博维诺模型动物园](https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models)

[OpenVino 模型下载](https://download.01.org/opencv/2019/open_model_zoo/)

[Github 上的 NCS 包装器源代码](https://github.com/tprlab/ncs-rest)