# 打开 STT 1.0 版本

> 原文：<https://towardsdatascience.com/open-stt-v10-release-1c28da67f2e6?source=collection_archive---------38----------------------->

我们终于成功了！

![](img/4caebc0cbb406f6f339b4f95d03b62b8.png)

*I wanted to give him an ushanka, but my paint editing skills are too poor*

# TLDR

这是一篇非常简短的关于**公开赛【STT / TTS v1.0** 的[发布](https://github.com/snakers4/open_stt/releases)的伴随帖。

**简单地说:**

*   **打开 STT** 在此[发布](https://github.com/snakers4/open_stt)，**打开 TTS** 在此[发布](https://github.com/snakers4/open_tts)，噪声数据集在此[发布](https://github.com/snakers4/asr-noises)；
*   我们在两个新的大型和多样化的领域中添加了两个新的数据集，大约有 **15，000 小时的注释**；
*   新数据集有**真实说话人标签**(即将发布)；
*   整体注释质量得到提高，大多数注释边缘案例得到修复；
*   数据集规范化大大提高；

# 打开 STT 摘要

前一段时间，我们对 STT 的总体状态(例如与计算机视觉相比)感到失望，尤其是在俄语方面。

总的来说，它面临着许多问题:(1)小/不太有用/不总是公开的学术数据集(2)巨大的不切实际的解决方案(3)庞大的/过时的/不切实际的工具包(4)一个有着悠久历史并已经被 SOTA 病毒困扰的行业(5)缺乏没有太多附加条件的可行解决方案。

所以我们[决定](https://spark-in.me/post/russian-open-stt-part1)从头开始为俄语建立一个数据集，然后基于数据集建立一套预先训练好的可部署模型。然后可能会涉及更多的语言。

开放 STT 可以说是目前存在的最大/最好的开放 STT / TTS 数据集。请点击以上链接了解更多信息。

# 最新版本中的主要功能

**重大最新** [**发布**](https://github.com/snakers4/open_stt/releases) **亮点:**

*   见上面 TLDR 的子弹；
*   新数据集的说话人标签；
*   数据集现在可以通过 torrent 以`.wav`文件的形式获得(我用 1 gbit/s 的通道播种)，也可以通过直接下载链接以`.mp3`文件的形式获得(下载速度也很快)；
*   覆盖 3 个主要领域的小型人工注释验证数据集(18 小时);
*   整体上游模型质量改进；
*   不再有“晃来晃去”的字母；
*   改进的语音活动检测；
*   极大地改进了数据集标准化；
*   显然注释并不完美，但是当我们添加新数据来过滤最讨厌的情况时，我们会不时地添加排除[列表](https://github.com/snakers4/open_stt/issues/5)；

**潜在用途:**

*   语音转文字(显然)；
*   去噪(为此还要考虑我们的`asr-noises` [数据集](https://github.com/snakers4/asr-noises))；
*   大规模文本到语音转换(**新增**)；
*   扬声器二进制化(**新增**)；
*   说话人识别(**新增**)；

# 批准

![](img/7dc631de3c5af1d2101c59986297fabd.png)

数据集大多是在`cc-nc-by` [许可](https://github.com/snakers4/open_stt/#license)下发布的。

如果您想将其用于商业用途，请提交一份[表格](https://forms.gle/nosMaNgj8MWKm99d9)，然后在这里联系我们[。](https://spark-in.me/cdn-cgi/l/email-protection#6906190c07361a1d1d290e06060e050c0e1b061c191a470a0604)

如果你需要一个快速(即不需要 GPU 运行)/可靠/离线 STT / TTS 系统，请联系[我](https://spark-in.me/cdn-cgi/l/email-protection#a1c0d7c4d8d2ced7e1c6ccc0c8cd8fc2cecc)。

# 下一步是什么？

*   改进/重新上传一些现有数据集，完善标签；
*   尝试用扬声器标签注释以前的数据；
*   发布预训练模型和后处理；
*   完善并发布扬声器标签；
*   大概是增加新的语言；
*   完善 STT 标签；

*原载于 2019 年 11 月 4 日*[*https://spark-in . me*](https://spark-in.me/post/open-stt-release-v10)*。*