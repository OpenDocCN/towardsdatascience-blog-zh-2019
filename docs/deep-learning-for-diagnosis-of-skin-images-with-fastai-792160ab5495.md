# 基于 fastai 的深度学习皮肤图像诊断

> 原文：<https://towardsdatascience.com/deep-learning-for-diagnosis-of-skin-images-with-fastai-792160ab5495?source=collection_archive---------4----------------------->

## 学会从皮肤镜图像中识别皮肤癌和其他疾病

我们展示了如何使用 fast.ai 解决 2018 年*皮肤病变分析对黑色素瘤检测*的挑战，并自动识别七种皮肤病变。

![](img/67900e6b19e7f0cc8bb23fcf5026fea2.png)

Pixabay/Pexels free images

由 Aldo von Wangenheim 发布— [aldo.vw@ufsc.br](mailto:aldo.vw@ufsc.br)

这是基于以下材料:

1.  [走向数据科学::用卷积神经网络对皮肤损伤进行分类](/classifying-skin-lesions-with-convolutional-neural-networks-fc1302c60d54)——医学深度学习指南和介绍
2.  Tschandl，Philipp，2018，“HAM10000 数据集，常见色素性皮肤病损的多源皮镜图像大集合”，[https://doi.org/10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T)，哈佛 data verse【arXiv 预印本:[arXiv:1803.10417](https://arxiv.org/abs/1803.10417)【cs。CV]]
3.  [用于处理 HAM10000 数据集的工具— GitHub](https://github.com/ptschandl/HAM10000_dataset) 。该存储库提供了对为组装训练数据集而创建和使用的工具的访问。

HAM10000 数据集(*具有 10000 幅训练图像的人机对抗*)用作 [ISIC 2018 挑战赛(任务 3)](http://arxiv.org/abs/1902.03368) 的训练集。通过挑战网站[https://challenge2018.isic-archive.com/](https://challenge2018.isic-archive.com/task3/)可以获得这个挑战的官方验证和测试集，没有地面真相标签。ISIC-Archive 还提供了一个[“现场挑战”提交网站](https://submission.challenge.isic-archive.com/#challenge/5bedea14c5eaea2c335effca)，用于在官方验证和测试集上连续评估自动分类器。

*   HAM10000 数据集也在 Kaggle 上。如果你想用 Kaggle 内核比较你的结果，看这里:[https://www . ka ggle . com/kmader/skin-cancer-mnist-ham 10000/kernels](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000/kernels)上次我看的时候(2019 年 6 月)，那里贴了 55 个不同的解决方案。
*   [本帖子中显示代码的笔记本(以及一些附加单元格)在这里](https://drive.google.com/open?id=1-6MlEi2sUW7AYRrBAjar_TsuoxsmUISo)。

这些图像看起来像这样:

![](img/1d83187b195fa9ec3cc13cd6ffe6b6c4.png)

Random images from the HAM10000 dataset with their ground truth labels

# 这个数据集中有什么？

该数据集包含通过标准皮肤镜检查获得的色素性皮肤损伤。这些是组织产生黑色素的病变，黑色素是人类皮肤的天然色素，并且是深色的。并非所有通过皮肤镜初步检查和分类的病变都一定是色素性病变。这意味着，在现实世界中，全科医生或护士通过皮肤镜检查患者(或患者进行自我检查)并打算将这些图像提交给皮肤科医生进行初步分类，可能会遇到除此数据集描述的病变之外的其他病变(参见*有 ISIC 2019 吗？*下)。

HAM10000 数据集中的病变类别包括:

1.  **nv** :黑素细胞痣——黑素细胞良性肿瘤[6705 张图片]
2.  **梅尔**:黑色素瘤——一种来源于黑色素细胞的恶性肿瘤【1113 张图片】
3.  **bkl** :良性角化病——包括脂溢性角化病、日光性雀斑样痣和扁平苔藓样角化病的一般类别【1099 张图片】；
4.  **基底细胞癌**:基底细胞癌——上皮性皮肤癌的一种常见变体，很少转移，但如果不治疗，会破坏性地生长(基底细胞癌不一定会产生色素性病变)[514 图片]；
5.  **akiec** :光化性角化病和上皮内癌——常见的非侵袭性鳞状细胞癌变体，无需手术即可局部治疗【327 张图片】；
6.  **vasc** :从樱桃血管瘤到血管角化瘤、化脓性肉芽肿的血管性皮肤病变【142 张图像】；
7.  **df** :皮肤纤维瘤——一种良性皮肤病变，被认为是良性增生或对轻微创伤的炎症反应【115 张图片】。

有关每个类的更详细描述，请查看 ka ggle kernel:[Skin Lesion Analyzer+tensor flow . js Web App-Python notebook 使用来自皮肤癌 MNIST: HAM10000](https://www.kaggle.com/vbookshelf/skin-lesion-analyzer-tensorflow-js-web-app) 的数据，以及上面 Philipp Tschandl 的论文。

# 这个数据集的诊断局限性是什么？

皮肤镜图像本身并不能为皮肤病学诊断或远程皮肤病学设置中可靠的远程患者分类提供足够的数据。皮肤镜图像缺乏上下文。为了提供背景信息，您需要执行一个图像采集协议，其中包括患者的全景全身图像和每个病灶的近似图像，这些图像是用尺子或图像中可见的其他参照系拍摄的，以便提供病灶大小的背景信息。用尺子拍摄的近似图像对于已经在治疗中的患者也很重要，以便允许随行医生跟踪病变的发展。为了正确获取全景图像和近似图像，需要遵循保证图像聚焦、从正确的距离拍摄并具有正确照明的协议来执行。还有一些细节不能通过目前使用的标准皮肤镜检查技术可靠地检测出来，在一些情况下，需要进行确认性活组织检查。

如果您有兴趣了解更多关于**远程皮肤病学检查采集协议**的信息，请点击此处:

*   奥尔多·冯·旺根海姆和丹尼尔·霍尔特豪森·努内斯。*创建支持临床协议和临床管理的网络基础设施:远程皮肤科的一个例子*。远程医疗和电子保健。提前上线:2018 年 11 月 30 日。【http://doi.org/10.1089/tmj.2018.0197】T4。在 ResearchGate 上也有一份[预印本。](https://www.researchgate.net/publication/329332581_Creating_a_Web_Infrastructure_for_the_Support_of_Clinical_Protocols_and_Clinical_Management_An_Example_in_Teledermatology)

# 皮肤镜图像是如何获得的？

现在使用的*接触式皮肤镜*是在 20 世纪 90 年代上半期由德国慕尼黑大学的一组研究人员领导的这种检查标准化的国际努力的结果。该设备使用放大 10 倍的单透镜，内部由发光二极管照明。使用矿物油进行检查，在将皮肤镜应用于病变部位并拍照之前，将矿物油应用于病变部位的表面。

![](img/754bd67680ec688698db4f52d553f58b.png)

Two different 10x contact dermoscopes: (a) analog pocket dermoscope and (b) dermoscopy adapter for Sony digital cameras employed by us at the [Santa Catarina State Telemedicine and Telehealth Network — STT/SC](http://site.telemedicina.ufsc.br/)

选择单眼 10 倍放大镜头作为标准允许开发非常小的设备，这很快变得非常流行。模拟皮肤镜可以放在胸前的口袋里，数字皮肤镜可以很容易地开发成小型 USB 设备或数码相机和智能手机的适配器。

该标准的缺点是 10 倍放大不足以可靠地检测某些病理，例如基底细胞癌，它是皮肤癌的最常见形式。这种形式的肿瘤以血管改变为特征，称为树枝状血管形成，不能用放大 10 倍的单目镜可靠地观察到:为了提供明确的诊断，必须进行确认性活检。可靠的检测需要更高的放大倍数和双目光学[1][2]。

![](img/538bfcc35ccdc147ba3704d86ddb3c35.png)

Non-pigmented basal cell carcinomas acquired (a) with a 10x contact planar dermoscope and (b) with a 50x binocular stereoscopic dermoscope, showing arboriform vascularizations (J.Kreusch, Univ. Lübeck, Sur Prise e.K. collection, POB 11 11 07, 23521 Lübeck, Germany) — only one image of the stereo pair is depicted here

直到 20 世纪 90 年代，还在开发其他类型的皮肤镜，它们可以提供更好的成像质量，但更大，不太容易操作，例如 5 *0x 双目立体接触皮肤镜*。这种设备更适合于病理的视觉早期检测，例如基底细胞癌[1][2]。然而，10 倍接触式单眼皮肤镜的实用性、普及性和标准化阻止了这些其他的研究方向。

![](img/d80e1fe5fcffd71db6c59c10d67f9f83.png)

50x sterescopic dermoscope produced by [Kocher Feinmechanik](https://www.kocher-feinmechanik.de/english/), Germany in the late 1990s and still in clinical use (L.F.Kopke, Florianópolis [2])

# 有没有 ISIC 2019？

[ISIC——国际皮肤成像协作组织](https://www.isic-archive.com/#!/topWithHeader/wideContentTop/main)已经纠正了 HAM10000 数据集中一些皮肤病诊断类别的缺乏，在 [ISIC 2019 挑战赛中发布了一个新的数据集:面向黑色素瘤检测的皮肤病变分析](https://challenge2019.isic-archive.com/)。2019 年 5 月 3 日发布的 2019 年数据集现在包含 9 个不同的诊断类别和 25，331 幅图像:

1.  黑素瘤
2.  黑素细胞痣
3.  基底细胞癌
4.  光化性角化病
5.  良性角化病(日光性雀斑样痣/脂溢性角化病/扁平苔藓样角化病)
6.  皮肤纤维瘤
7.  血管病变
8.  鳞状细胞癌
9.  其他人都没有

*   在撰写本文时，这个新数据集的测试元数据还不可用。宣布将于 2019 年 8 月 9 日上映。
*   到目前为止，ISIC 已经赞助了图像分析方面的四项挑战:ISIC 2016 至 ISIC 2019，始终以“皮肤病变分析促进黑色素瘤检测”为主题。[这四项挑战可在国际标准行业分类档案中找到](https://www.isic-archive.com/#!/topWithHeader/tightContentTop/challenges)。

让我们开始研究 2018 年挑战的数据集。

# 初始化

每次运行此[笔记本](https://drive.google.com/open?id=1-6MlEi2sUW7AYRrBAjar_TsuoxsmUISo)时，请执行以下本节中的操作一次…

```
%reload_ext autoreload
%autoreload 2
%matplotlib inline
```

# 在 Google Colab 上测试您的虚拟机…

只是为了确定，看看哪一个 CUDA 驱动和哪一个 GPU Colab 已经为你提供了。GPU 通常是:

*   一个 11 GB 内存的 K80 或者(如果你真的幸运的话)
*   一辆配有 14 GB 内存的特斯拉 T4

如果谷歌的服务器很拥挤，你最终只能访问 GPU 的一部分。如果你的 GPU 与另一台 *Colab* 笔记本共享，你会看到可供你使用的内存量减少。

小贴士:避开美国西海岸的高峰期。我住在 GMT-3，我们比美国东海岸早两个小时，所以我总是试图在早上进行繁重的处理。

```
!/opt/bin/nvidia-smi
!nvcc --version
```

当我开始运行这里描述的实验时，我很幸运:我有一个 15079 MB RAM 的完整 T4！我的输出如下所示:

```
Thu May  2 07:36:26 2019       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.79       Driver Version: 410.79       CUDA Version: 10.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   63C    P8    17W /  70W |      0MiB / 15079MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Sat_Aug_25_21:08:01_CDT_2018
Cuda compilation tools, release 10.0, V10.0.130
```

# 图书馆进口

在这里，我们导入所有必需的包。我们将与 [fastai V1 库](http://www.fast.ai/2018/10/02/fastai-ai/)合作，它位于 [Pytorch 1.0](https://hackernoon.com/pytorch-1-0-468332ba5163) 之上。fastai 库提供了许多有用的功能，使我们能够快速轻松地构建神经网络并训练我们的模型。

```
from fastai.vision import *
from fastai.metrics import error_rate
from fastai.callbacks import SaveModelCallback# Imports for diverse utilities
from shutil import copyfile
import matplotlib.pyplot as plt
import operator
from PIL import Image
from sys import intern   # For the symbol definitions
```

# 导出和恢复功能

```
# Export network for deployment and create a copydef exportStageTo(learn, path):
    learn.export()
    # Faça backup diferenciado
    copyfile(path/'export.pkl', path/'export-malaria.pkl')

#exportStage1(learn, path)# Restoration of a deployment model, for example in order to conitnue fine-tuningdef restoreStageFrom(path):
  # Restore a backup
  copyfile(path/'export-malaria.pkl', path/'export.pkl')
  return load_learner(path)

#learn = restoreStage1From(path)
```

# 下载色素性病变的皮肤镜图像

我们将下载这个数据集的 Kaggle 版本，因为 Google Colab 已经预装了 Kaggle API，并且都组织在一个版本中。zip 文件。为了从 Kaggle 下载，您需要:

*   卡格尔的一个账户
*   要在 Colab 上安装您的 Kaggle 凭证(一个. json 文件)

要了解如何做到这一点，首先查看本教程和 Kaggle API 说明，然后生成您的凭证并上传到 Colab:

*   [TowardsDataScience::在 Google Colab 中设置 ka ggle——Anne Bonner 为新手编写的简单教程](/setting-up-kaggle-in-google-colab-ebb281b61463)
*   [https://github.com/Kaggle/kaggle-api#api-credentials](https://github.com/Kaggle/kaggle-api#api-credentials)

# 当你创建并复制你的 Kaggle 证书到 Colab

运行下面的单元格。它将为您的 Kaggle API 凭据创建一个文件夹，并将您的凭据安装在 Colab:

```
!mkdir .kaggle
!mv kaggle.json .kaggle
!chmod 600 /content/.kaggle/kaggle.json
!cp /content/.kaggle/kaggle.json ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!kaggle config set -n path -v{/content}
```

出于某种神秘的原因，这个脚本有时不工作。这似乎与 Colab 命名主文件夹的形式有关。如果您遇到错误消息，只需再次执行它。最终输出应该如下所示:

```
- path is now set to: {/content}
```

# 执行实际的下载和解压缩

创建一个“数据”文件夹，并将皮肤镜检查图像下载到其中

```
!mkdir data
!kaggle datasets download kmader/skin-cancer-mnist-ham10000 -p data
```

这将产生以下输出:

```
Downloading skin-cancer-mnist-ham10000.zip to data
100% 2.61G/2.62G [00:52<00:00, 42.3MB/s]
100% 2.62G/2.62G [00:52<00:00, 53.4MB/s]
```

将整个 zip 文件解压到/content/data 中，然后悄悄地(-q)解压图像文件(你不想冗长地解压超过 10k 的图像吧！).我们将使用 override 选项(-o)，以便允许安静地覆盖那些在您之前中断的尝试中创建的文件。

```
# Unzip the whole zipfile into /content/data
!unzip -o data/skin-cancer-mnist-ham10000.zip -d data
# Quietly unzip the image files
!unzip -o -q data/HAM10000_images_part_1.zip -d data
!unzip -o -q data/HAM10000_images_part_2.zip -d data
# Tell me how many files I unzipped///
!echo files in /content/data: `ls data | wc -l`
```

如果您有 10，023 个文件，那么您就做对了！

```
Archive:  data/skin-cancer-mnist-ham10000.zip
  inflating: data/hmnist_28_28_RGB.csv  
  inflating: data/HAM10000_metadata.csv  
  inflating: data/HAM10000_images_part_1.zip  
  inflating: data/hmnist_28_28_L.csv  
  inflating: data/hmnist_8_8_L.csv   
  inflating: data/HAM10000_images_part_2.zip  
  inflating: data/hmnist_8_8_RGB.csv  
files in /content/data: 10023
```

# 准备您的数据

现在，我们将准备使用 fast.ai 处理我们的数据。

在我们之前的医学图像分类帖子([利用 fastai](/deep-learning-and-medical-image-analysis-for-malaria-detection-with-fastai-c8f08560262f) 进行疟疾检测的深度学习和医学图像分析)中，我们已经将图像类别分类到文件夹中，每个类别一个文件夹。

在这里，我们将所有图像存储在一个文件夹中，并将元数据存储在一个电子表格中，我们将使用 fast.ai 数据块 API 中的 fast . ai*imagedata bunch . from _ CSV()*方法读取该电子表格。

# 我们这里有什么不同？

HAM10000 数据集不提供根据类别分类到文件夹中的图像。相反，所有图像都在一个文件夹中，并且提供了一个电子表格，其中包含每个图像的若干元数据。在本教程中，我们将从这里读取每个图像的类。csv 电子表格，而不是将图像文件组织到文件夹中，其中文件夹的名称是图像所属的类。 **fast.ai** 还提供了解释电子表格和提取图像分类数据的现成方法。在这篇文章中，我们将学习如何利用这些方法。

为此，我们将使用 fast.ai 的[数据块 API](https://docs.fast.ai/data_block.html) 。在下面的帖子中有一个很好的解释:

*   Tom McKenzie 的 [Medium::使用 fastai 数据块 API](https://medium.com/@tmckenzie.nz/using-the-fastai-data-block-api-b4818e72155b)

# 创建您的培训和验证数据集

在上面使用 Keras 的原始教程中，有一个从数据中创建训练、验证和测试文件夹的例程。有了 fast.ai 就不需要了:如果你只有一个“train”文件夹，你可以在创建 DataBunch 的时候通过简单的传递几个参数来拆分它…

通过 fast.ai，我们还可以轻松地使用不同于原始 ImageNet 分辨率的分辨率，这些分辨率用于预先训练我们将使用的网络。在上面列出的教程中，作者将数据集图像分辨率降低到 224x224，以便使用 Keras MobileNet 模型。我们将采用 448x448 的分辨率:

```
bs = 64        # Batch size, 64 for medium images on a T4 GPU...
size = 448      # Image size, 448x448 is double than the orignal 
                # ImageNet
path = Path("./data")   # The path to the 'train' folder you created...# Limit your augmentations: it's medical data! You do not want to phantasize data...
# Warping, for example, will let your images badly distorted, so don't do it!
# This dataset is big, so don't rotate the images either. Lets stick to flipping...
tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_rotate=None, max_warp=None, max_zoom=1.0)
# Create the DataBunch!
# Remember that you'll have images that are bigger than 128x128 and images that are smaller,   
# so squish them all in order to occupy exactly 128x128 pixels...
data = ImageDataBunch.from_csv('data', csv_labels='HAM10000_metadata.csv', suffix='.jpg', fn_col=1, label_col=2, 
                               ds_tfms=tfms, valid_pct = 0.2,size=size, bs=bs)
print('Transforms = ', len(tfms))
# Save the DataBunch in case the training goes south... so you won't have to regenerate it..
# Remember: this DataBunch is tied to the batch size you selected. 
data.save('imageDataBunch-bs-'+str(bs)+'-size-'+str(size)+'.pkl')
# Show the statistics of the Bunch...
print(data.classes)
data
```

这将产生以下输出:

```
Transforms =  2
['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']ImageDataBunch;Train: LabelList (8012 items)
x: ImageList
Image (3, 448, 448),Image (3, 448, 448),Image (3, 448, 448),Image (3, 448, 448),Image (3, 448, 448)
y: CategoryList
bkl,bkl,bkl,bkl,bkl
Path: data;Valid: LabelList (2003 items)
x: ImageList
Image (3, 448, 448),Image (3, 448, 448),Image (3, 448, 448),Image (3, 448, 448),Image (3, 448, 448)
y: CategoryList
bkl,nv,nv,nv,nv
Path: data;Test: None
```

# 查看您的数据集群，看看增加是否可以接受…

```
data.show_batch(rows=5, figsize=(15,15))
```

![](img/1d83187b195fa9ec3cc13cd6ffe6b6c4.png)

# 第一次训练实验:ResNet34

如果您不知道使用什么模型，从 34 层的剩余网络开始是一个不错的选择。功能强大，但不太小也不太大…

在上面列出的教程中，作者使用了在 Keras 中实现的 MobileNet 和网络的原始图像分辨率 224x224。在 fast.ai 中，ResNet 很容易适应我们的 DataBunch 的 448x448 分辨率。

现在我们将开始训练我们的模型。我们将使用一个[卷积神经网络](http://cs231n.github.io/convolutional-networks/)主干和一个单隐层全连接头作为分类器。不知道这些东西是什么意思？不要担心，我们将在接下来的课程中深入探讨。目前，您需要知道我们正在构建一个模型，该模型将图像作为输入，并将输出每个类别的预测概率(在这种情况下，它将有 37 个输出)。

我们将使用两种不同的指标来衡量我们的培训成功程度:

*   **精度**:验证精度
*   **错误率**:验证错误率

如果你想了解更多信息，请看 https://docs.fast.ai/metrics.html 的。

```
learn = cnn_learner(data, models.resnet34, metrics=[accuracy, error_rate, dice(iou=True), fbeta])
learn.model
```

只需将包含 *DataBunch* 实例的 *data* 变量传递给 *cnn_learner()* 函数，fast.ai 就会自动调整新网络的输入层以适应更高的图像分辨率。该模型将如下所示:

```
Sequential(
  (0): Sequential(
    (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace)
    (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (4): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (2): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (5): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
...
...
... and so on...
```

# 培训策略

我们将采用 Leslie N. Smith 开发的 *fit1cycle* 方法，详情见下文:

*   [https://docs.fast.ai/callbacks.one_cycle.html](https://docs.fast.ai/callbacks.one_cycle.html)
*   ***一种训练有素的神经网络超参数方法:第一部分——学习速率、批量大小、动量和权重衰减*【https://arxiv.org/abs/1803.09820】T21**
*   ***超收敛:使用大学习率非常快速地训练残差网络***—[https://arxiv.org/abs/1708.07120](https://arxiv.org/abs/1708.07120)
*   还有一篇来自 [Nachiket Tanksale](https://towardsdatascience.com/@nachiket.tanksale) 的非常有趣的文章，名为[寻找好的学习率和单周期政策](/finding-good-learning-rate-and-the-one-cycle-policy-7159fe1db5d6)，其中讨论了周期学习率和动量
*   因为这种方法很快，所以在第一个迁移学习阶段，我们将只使用 10 个时期

如果你想了解更多关于 fast.ai 库中新的学习 API 的信息，可以看看 Sylvain Gugger 准备的这个笔记本。

如果性能变得更好，我们也将在每个时期保存网络:[https://docs.fast.ai/callbacks.html#SaveModelCallback](https://docs.fast.ai/callbacks.html#SaveModelCallback)

```
learn.fit_one_cycle(10, callbacks=[SaveModelCallback(learn, every='epoch', monitor='accuracy', name='derma-1')])
# Salve a rede (necessita regerar o databunch caso a gente continue)
learn.save('derma-stage-1')
# Faça o deploy desta rede para podermos usar offline depois para fazer testes
exportStageTo(learn, path)
```

![](img/2543f47b273105ccc3bab6d0c2c62167.png)

85% accuracy…

# ResNet34 的结果

让我们看看我们得到了什么结果。

我们将首先看到模型最容易混淆的类别。我们将尝试看看模型预测的结果是否合理。在这种情况下，错误看起来是合理的(没有一个错误看起来明显幼稚)。这表明我们的分类器工作正常。

此外，当我们绘制混淆矩阵时，我们可以看到分布严重偏斜:该模型一次又一次地犯同样的错误，但它很少混淆其他类别。这表明它只是发现很难区分彼此之间的某些特定类别；这是正常的行为。

让我们生成一个*分类解释*，看看一些结果，混淆矩阵和损失曲线。

```
interp = ClassificationInterpretation.from_learner(learn)losses,idxs = interp.top_losses()len(data.valid_ds)==len(losses)==len(idxs)
```

查看您的最差结果，首先不使用热图:

```
interp.plot_top_losses(9, figsize=(20,11), **heatmap=False**)
```

![](img/1bd954979c33055b98a4f0180d736eee.png)

现在，做同样的事情，但是用热图突出显示图像，以便查看每个图像的哪些部分导致了错误的分类:

```
interp.plot_top_losses(9, figsize=(20,11), **heatmap=True**)
```

![](img/b20a35bac6594847e352481139756fbb.png)

# 显示混淆矩阵

这里我们有 ***七个职业*** 看看混乱矩阵很有意义。此外，它能拍出漂亮的照片…

```
interp.plot_confusion_matrix(figsize=(5,5), dpi=100)
```

![](img/e27620344ba06bf76e1f6b39543080b5.png)

我们在这里可以看到:

*   痣是最常见的现象。人们可以考虑减少训练集中的痣的实例，以便不扭曲结果；
*   有几种良性角化病( **bkl** )被错误分类。这可能是因为该数据集中的 ***bkl*** 是一个通用类别，包括脂溢性角化病、日光性雀斑样痣和扁平苔藓样角化病，这些皮肤病即使相关，看起来也非常不同；
*   还有几个黑色素瘤( **mel** )被错误分类。这是一个惊喜。我期望网络在这里表现得更好。

如果你对混淆矩阵感到困惑，请看这里:

*   [从混乱矩阵中找出混乱](/taking-the-confusion-out-of-confusion-matrices-c1ce054b3d3e)作者[艾莉森·拉冈](https://towardsdatascience.com/@allison.n.ragan)
*   [通过](https://medium.com/datadriveninvestor/simplifying-the-confusion-matrix-aa1fa0b0fc35)[马达夫·米什拉](https://medium.com/@madhav.mishra)简化混淆矩阵

# 展示你的学习曲线:

画出你的损失，以便看到学习曲线:

```
learn.recorder.plot_losses()
```

![](img/edfaa89c8677179a6dd33e7dfc658441.png)

这个成绩确实不错。网络有些波动，但学习稳定。现在让我们试着微调网络。

# 微调 ResNet34

首先解冻网络，并尝试为这个特定的网络找到一个好的学习速率范围。

方法 [*learn.lr_find()*](https://docs.fast.ai/callbacks.lr_finder.html) 帮你找到一个最优的学习速率。它使用了 2015 年论文*中开发的技术，用于训练神经网络*([http://arxiv.org/abs/1506.01186](http://arxiv.org/abs/1506.01186))的循环学习率，我们只是从一个非常小的值开始增加学习率，直到损失开始减少。

如果你想知道更多关于寻找最佳学习率的信息，请看这里:

*   [**迈向数据科学** :: *用 LR-Finder 加速神经网络训练——为你的网络找到良好的初始学习率*faiz an Ahemad](/speeding-up-neural-net-training-with-lr-finder-c3b401a116d0)

让我们开始吧:

```
# Unfreeze the network
learn.unfreeze()
# Find optimum learning rates
learn.lr_find()
# Include **suggestion=True** in order to obtain a suggestion on where to look...
learn.recorder.plot(suggestion=True)
```

让我们微调一下结果。为了确保万无一失，我们将使用 30 个纪元。学习率搜索器将 1e-5 确定为“安全”学习率。因此，我们将使用*经验法则*定义一个*学习速率范围*:以“安全”速率 1e-5 结束，以高一个数量级的速率开始: **max_lr=slice(1e-4，1e-5)** 。

```
# Unfreeze the network
learn.unfreeze()learn.fit_one_cycle(30, **max_lr=slice(1e-4,1e-5)**, 
                    callbacks=[SaveModelCallback(learn, every='epoch', monitor='accuracy', name='derma')])
# Agora, salve como estágio 2...
learn.save('derma-stage-2')
# Deploy definitivo
exportStageTo(learn, path)
```

![](img/df7c8d3dd1676e7c46510b3b863875f2.png)

93% accuracy!

所以我们第一次运行就达到了 93%的准确率。这很好！在上面的教程和内核中达到的准确率分别是 85%和 86%。

现在让我们看看我们的统计数据。：

```
interp = ClassificationInterpretation.from_learner(learn)losses,idxs = interp.top_losses()# Test to see if there's not anything missing (must return True)
len(data.valid_ds)==len(losses)==len(idxs)
```

如果返回 True，则绘制微调网络的混淆矩阵:

```
interp.plot_confusion_matrix(figsize=(5,5), dpi=100)
```

![](img/262196962eb4e86860e19f35c6f10305.png)

在这个矩阵中黑色素瘤的预测看起来更好！让我们看看训练曲线:

```
learn.recorder.plot_losses()
```

![](img/b3da65a1f2dfef404301798357cb8e94.png)

我们可以看到，训练和验证曲线都在振荡。训练曲线似乎到达了一个平台，并且与验证曲线分离，验证曲线下降得更慢。这表明我们可能正在走向网络的过度拟合。它会被指示到此为止。

在 Colab 网站上我们提供的[笔记本中，我们又对网络进行了 30 个纪元的训练，以确保万无一失。它实际上变得更糟，可能是由于过度拟合。所以停在这里对 ResNet34 来说是个不错的选择。](https://drive.google.com/open?id=1-6MlEi2sUW7AYRrBAjar_TsuoxsmUISo)

# 变大:ResNet50

ResNet34 的准确率达到了 92.9%。让我们看看更大的网络能否让我们表现得更好。我们将再次创建 DataBunch，这次使用较小的批处理大小，以避免 GPU 内存过载…

```
bs = 28         # Batch size, 28 for medium images on a T4 GPU and ResNet50...
size = 448      # Image size, 448x448 is double than the orignal 
                # ImageNet size of the pre-trained ResNet we'll be using, 
                # should be easy to train...
path = Path("./data")   # The path to the 'train' folder you created...# Limit your augmentations: it's medical data! You do not want to phantasize data...
# Warping, for example, will let your images badly distorted, so don't do it!
# This dataset is big, so don't rotate the images either. Lets stick to flipping...
tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_rotate=None, max_warp=None, max_zoom=1.0)
# Create the DataBunch!
# Remember that you'll have images that are bigger than 128x128 and images that are smaller,   
# so squish them all in order to occupy exactly 128x128 pixels...
data = ImageDataBunch.from_csv('data', csv_labels='HAM10000_metadata.csv', suffix='.jpg', fn_col=1, label_col=2, 
                               ds_tfms=tfms, valid_pct = 0.2,size=size, bs=bs)
print('Transforms = ', len(tfms))
# Save the DataBunch in case the training goes south... so you won't have to regenerate it..
# Remember: this DataBunch is tied to the batch size you selected. 
data.save('imageDataBunch-bs-'+str(bs)+'-size-'+str(size)+'.pkl')
# Show the statistics of the Bunch...
print(data.classes)
data
```

现在输出应该是这样的:

```
Transforms =  2
['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']ImageDataBunch;Train: LabelList (8012 items)
x: ImageList
Image (3, 448, 448),Image (3, 448, 448),Image (3, 448, 448),Image (3, 448, 448),Image (3, 448, 448)
y: CategoryList
bkl,bkl,bkl,bkl,bkl
Path: data;Valid: LabelList (2003 items)
x: ImageList
Image (3, 448, 448),Image (3, 448, 448),Image (3, 448, 448),Image (3, 448, 448),Image (3, 448, 448)
y: CategoryList
nv,nv,nv,mel,bkl
Path: data;Test: None
```

现在创建一个 ResNet50:

```
learn50 = cnn_learner(data, models.resnet50, metrics=[accuracy, error_rate])
learn50.model
```

请注意，我们在指标中包括了准确性，因此我们不需要根据错误率手动执行计算。模型看起来应该是这样的:

```
Sequential(
  (0): Sequential(
    (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace)
    (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (4): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
        (downsample): Sequential(
          (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
      )
      (2): Bottleneck(
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
      )
    )
...
...
...
(1): Sequential(
    (0): AdaptiveConcatPool2d(
      (ap): AdaptiveAvgPool2d(output_size=1)
      (mp): AdaptiveMaxPool2d(output_size=1)
    )
    (1): Flatten()
    (2): BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.25)
    (4): Linear(in_features=4096, out_features=512, bias=True)
    (5): ReLU(inplace)
    (6): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): Dropout(p=0.5)
    (8): Linear(in_features=512, out_features=7, bias=True)
  )
)
```

把它学会:

```
learn50.fit_one_cycle(15, callbacks=[SaveModelCallback(learn50, every='epoch', monitor='accuracy', name='derma50-1')])
# Save weights
learn50.save('derma50-stage-1')
# Deploy the whole network (with the databunch)
exportStageTo(learn50, path)
```

![](img/4600f60dca5bc6dc51f1da4e96c7e0b0.png)

87.6% accuracy….

看看结果:

```
interp = ClassificationInterpretation.from_learner(learn50)losses,idxs = interp.top_losses()interp.plot_confusion_matrix(figsize=(5,5), dpi=100)
```

![](img/607e886f7bf14481f83290eb0ec66931.png)

看看学习曲线:

```
learn50.recorder.plot_losses()
```

![](img/19b30d8fa4ced7b891cb74004b880141.png)

目前，这并没有给人留下深刻印象:在第一个迁移学习阶段，网络比 ResNet34 振荡得更多，结果即使在数字上更好(85% x 87%的准确率)，实际上在混淆矩阵的视觉分析上看起来更差。让我们对 ResNet50 进行微调，看看这是否会产生更好的结果。

我们将在两个实验中完成这项工作。

# ResNet50 实验#1:盲接受学习速率建议的微调

当训练一个深度神经网络时，选择一个好的学习速率对于快速收敛和较低的误差都是至关重要的。微调第一步是找到一个合适的学习速率范围:

```
# Unfreeze the network
learn50.unfreeze()
# Find optimum learning rates
learn50.lr_find()
# Include suggestion=True in order to obtain a suggestion on where to look...
learn50.recorder.plot(suggestion=True)
```

这将输出:

```
LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.
Min numerical gradient: 9.12E-07
Min loss divided by 10: 1.10E-07
```

![](img/7e94eb56b6c07179eb44bd53364d07ee.png)

下限，9.12E-07，我们就上舍入到 1.0E-06。现在，您可以使用我们的经验法则进行微调。我们将使用*经验法则*定义学习率的*范围；以“安全”速率 1e-06 结束；以高一个数量级的速率开始；*

```
# Unfreeze the network
learn50.unfreeze()
learn50.fit_one_cycle(35, **max_lr=slice(1e-5,1e-6)**, 
                    callbacks=[SaveModelCallback(learn50, every='epoch', monitor='accuracy', name='derma50')])# Save the weights of stage 2 each "better" epoch:
learn50.save('derma50-stage-2')
# **Do not** overwrite the stage 1 .pkl with  stage 2 
# We will need it for the ResNet50 Experiment #2
# exportStageTo(learn50, path)
```

![](img/270e55fa5f23a7ade7b29e8d8f519ec7.png)

90% accuracy…

所以，经过 35 个时期和大量的处理，我们达到了 90%的准确率。这看起来没有希望…让我们看看困惑矩阵和学习曲线:

```
interp = ClassificationInterpretation.from_learner(learn50)losses,idxs = interp.top_losses()interp.plot_confusion_matrix(figsize=(5,5), dpi=100)
```

![](img/a1c70d442140d47d8b4fe8bc0135d26a.png)

```
learn50.recorder.plot_losses()
```

![](img/506d6e2c0fda6bf545758b914dd4b359.png)

我们在这里看到的是一个与 ResNet34 非常不同的结果。对于 ResNet50，网络在微调阶段学习训练集，但它一直在振荡:一些批次使其性能更好，另一些批次使其返回并性能更差。这通常意味着数据质量差，噪音太多。然而，在这种情况下，我们已经看到，我们可以使用 ResNet34 实现 93%的准确性。所以，坏数据不是这里的情况。另一种可能性是，网络有太多的参数，它不是泛化，而是适应训练集的单个实例，从而学习单个实例并去泛化。对于数据集的其他部分，这使得它的性能更差，因此网络就像一个钟摆，在误差空间中来回摆动。验证损失比整个微调过程中的训练损失高得多，证实了对学习曲线的这种解释。

但是我们很固执…让我们做另一个实验。

# ResNet50 实验#2:使用手动学习率设置进行微调

在以前的实验中，我们盲目地接受了分析算法的建议，并采用了非常低的学习率。也许这就是学习不好的原因？

如果我们看一下学习率图，我们可以看到一个 ***平台*** 已经在大约 1.0E-4 处形成。然后学习率陷入两个低谷，一个在 1.0E-5，另一个在 1.0E-6。如果我们采用更稳定的、平坦的区域，从 1.0E-4 到 1.0E-5，并把它作为我们的学习率范围，会怎么样？

首先，将网络恢复到完成初始迁移学习时的状态:

```
# Will always load a *path/'export.pkl'* deployment file
learn50 = restoreStageFrom(path)
```

再次微调，现在用 **max_lr=slice(1e-4，1e-5)** :

```
# Unfreeze the network
learn50.unfreeze()
learn50.fit_one_cycle(35, **max_lr=slice(1e-4,1e-5)**, 
                    callbacks=[SaveModelCallback(learn50, every='epoch', monitor='accuracy', name='derma50')])
learn50.save('derma50-stage-2')
exportStageTo(learn50, path)
```

![](img/46b61931d42a09b3e9a4446189cd9d34.png)

92.6% accuracy….

让我们来看看结果图:

![](img/c820f5c4403499133838f78f3767a04e.png)![](img/09b84b530c9dbfce7efddbb3a65515bd.png)

这些结果比以前好，即使网络振荡很大。然而，混淆矩阵向我们显示，黑色素瘤*(这里最重要的病理)的结果并不乐观。*

*让我们反其道而行之，尝试一个比 ResNet34 小的网络，而不是比 ResNet18 大的。*

# *ResNet18*

*那么，为什么不尝试一个小得多的网络，看看它的表现呢？让我们用皮肤镜数据来试一试。*

*为此，我们将更改批量大小并重新生成数据分组。ResNet18s 要小得多，我们将有更多的可用内存，因此使用更大的批处理大小是有意义的:*

```
*bs = 48         # Batch size, 64 for medium images on a T4 GPU and ResNet18...
size = 448      # Image size, 448x448 is double than the orignal 
                # ImageNet size of the pre-trained ResNet we'll be using, 
                # should be easy to train...
path = Path("./data")   # The path to the 'train' folder you created...# Limit your augmentations: it's medical data! You do not want to phantasize data...
# Warping, for example, will let your images badly distorted, so don't do it!
# This dataset is big, so don't rotate the images either. Lets stick to flipping...
tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_rotate=None, max_warp=None, max_zoom=1.0)
# Create the DataBunch!
# Remember that you'll have images that are bigger than 128x128 and images that are smaller,   
# so squish them all in order to occupy exactly 128x128 pixels...
data = ImageDataBunch.from_csv('data', csv_labels='HAM10000_metadata.csv', suffix='.jpg', fn_col=1, label_col=2, 
                               ds_tfms=tfms, valid_pct = 0.2,size=size, bs=bs)
print('Transforms = ', len(tfms))
# Save the DataBunch in case the training goes south... so you won't have to regenerate it..
# Remember: this DataBunch is tied to the batch size you selected. 
data.save('imageDataBunch-bs-'+str(bs)+'-size-'+str(size)+'.pkl')
# Show the statistics of the Bunch...
print(data.classes)
data*
```

*创建网络:*

```
*learn = cnn_learner(data, models.resnet18, metrics=[accuracy, error_rate])
learn.model*
```

*该模型将如下所示(注意，这个较小的 ResNet 模型只有 1024 个 out 特性，而不是最后一个块中的 4096 个):*

```
*Sequential(
  (0): Sequential(
    (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace)
    (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (4): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
...
...
...
  (1): Sequential(
    (0): AdaptiveConcatPool2d(
      (ap): AdaptiveAvgPool2d(output_size=1)
      (mp): AdaptiveMaxPool2d(output_size=1)
    )
    (1): Flatten()
    (2): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.25)
    (4): Linear(in_features=1024, out_features=512, bias=True)
    (5): ReLU(inplace)
    (6): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): Dropout(p=0.5)
    (8): Linear(in_features=512, out_features=7, bias=True)
  )
)*
```

*让我们转移-学习 15 个时代:*

```
*learn.fit_one_cycle(15, callbacks=[SaveModelCallback(learn, every='epoch', monitor='accuracy', name='derma-1')])
learn.save('derma-stage-1')
exportStageTo(learn, path)*
```

*![](img/e2845a99fef77f73157199243b6c8e8e.png)*

*85.5% accuracy…*

```
*interp = ClassificationInterpretation.from_learner(learn)losses,idxs = interp.top_losses()interp.plot_confusion_matrix(figsize=(5,5), dpi=100)*
```

*![](img/63df78bc313d16811d57669f8da837e1.png)*

```
*learn.recorder.plot_losses()*
```

*![](img/388f0ffd5b2a9a53e33ce30218db6703.png)*

*从数值上看，这比 ResNet34 好 0.5%。让我们看看它在微调后的表现:*

```
*# Unfreeze the network
learn.unfreeze()
# Find optimum learning rates
learn.lr_find()
# Include suggestion=True in order to obtain a suggestion on where to look...
learn.recorder.plot(suggestion=True)*
```

*同样，我们有一条曲线，首先达到一个平台，然后陷入两个洞。只是这里的洞比高地更深。让我们考虑，在这种情况下，空洞是显著的，并接受学习率查找器的建议，使 **max_lr=slice(1e-5，1e-6)** :*

```
*LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.
Min numerical gradient: 1.32E-06
Min loss divided by 10: 1.58E-07*
```

*![](img/863b103759225a33b9258a169f51ec7d.png)*

```
*# Unfreeze the network
learn.unfreeze()
learn.fit_one_cycle(35, **max_lr=slice(1e-5,1e-6)**, 
                    callbacks=[SaveModelCallback(learn, every='epoch', monitor='accuracy', name='derma18')])learn.save('derma18-stage-2')exportStageTo(learn, path)*
```

*![](img/6eae1c1925923a56b8c1aaf35f3186e0.png)*

*88%的准确率！那比 ResNet34 差多了。让我们绘制学习曲线:*

```
*learn.recorder.plot_losses()*
```

*![](img/92c1e2677113a0a84ceda2500fcabf04.png)*

*我们在这里看到的是，当我们以太小的学习率训练它时，ResNet18 比 ResNet50 振荡得更多。这表明这里的学习率也太小了。*

*我重复了我们之前对 ResNet50 进行的微调实验#2，并将学习率范围设置为 **max_lr=slice(1e-4，1e-5)** ，并再次训练它。在这种情况下，ResNet18 实现了 0.904643 的验证精度。结果图形如下所示:*

*![](img/77582c5b64e0a941ec43d66ad9f908fa.png)**![](img/7e3cc91c3a77658b08d33851d7f3df99.png)*

*这比以前好，但仍比 ResNet34 和 ResNet50 差。ResNet18 似乎不是解决这个问题的好选择。[代码在我们的笔记本里](https://drive.google.com/open?id=1-6MlEi2sUW7AYRrBAjar_TsuoxsmUISo)。*

# *如果培训中途中断，我该怎么办？*

*如果你的训练被打断了，你会怎么做？这可能是因为你在 Google Colab 笔记本上达到了连续 12 小时的“免费”操作时间，或者因为你的计算机由于某种原因停止了。我住在巴西，电力短缺是常事…*

**fit_one_cycle* 方法适用于变化的自适应学习速率，遵循速率先增大后减小的曲线。如果你中断第 10 个纪元的训练，比如说 20 个纪元，然后重新开始 9 个以上的纪元，**你将不会得到与不间断训练 20 个纪元**相同的结果。您必须能够记录您停止的位置，然后从该点重新开始训练周期，并使用该周期部分的正确超参数。*

*![](img/8d5fa3658ed88e31179157e295612d77.png)*

*A fit_one_cycle training session divided into three subsessions. Image by PPW@GitHub*

*你要做的第一件事就是保存你的网络:*

```
*learn.fit_one_cycle(20, max_lr=slice(1e-5,1e-6), 
       callbacks=[SaveModelCallback(learn, every='epoch',  
                  monitor='accuracy', name='saved_net')])*
```

*这将使您的网络在每个纪元都被保存，您提供的名称后面跟着*_ #纪元*。所以在纪元#3，文件 *saved_net_3.pth* 将被写入。您可以在完成以下操作后加载此文件:*

*   *重新创建了*数据束*和*
*   *用它重新实例化了网络。*

*重装完*后。pth* 文件，你可以重新开始你的训练，只是你要告诉 *fit_one_cycle* 考虑 20 个历元，但是要从历元#4 开始训练。*

*要了解这是如何做到的，请看这里:*

*   *[把一个长周期的政策分成几个小周期——PPW 的 GitHub](https://github.com/PPPW/deep-learning-random-explore/tree/master/divide_1cycle?source=post_page---------------------------)*

***你是怎么做到的？***

*fast.ai 中的 *fit_one_cycle* 方法已经开发出来，允许您告诉它从周期的哪个部分恢复中断的训练。恢复培训的代码如下所示:*

```
*# Create a new net if training was interrupted and you had to 
# restart your Colab sessionlearn = cnn_learner(data, models.<your_model_here>, 
                    metrics=[accuracy, error_rate])# If you're resuming, only indicating the epoch from which to 
# resume, indicated by ***start_epoch=<epoch#>*** will load the last 
# saved .pth, it is not necessary to explicitly reload the last 
# epoch, you only should **NOT** change the name given in 
# name=<callback_save_file>:
# when resuming fast.ai will try to reload 
# ***<callback_save_file>_<previous_epoch>.pth***
# Unfreeze the network
learn50.unfreeze()# Use start_epoch=<some_epoch> to resume training...
learn.fit_one_cycle(20, max_lr=slice(1e-5,1e-6), 
                    ***start_epoch=<next_epoch#>***,
                    callbacks=[SaveModelCallback(learn, 
                    every='epoch', monitor='accuracy', 
                    ***name=<callback_save_file>***)])*
```

*fast.ai 会告诉你“*载入<回调 _ 保存 _ 文件> _ <上一个 _ 纪元# >* ”，恢复训练。*

*您可以在此查看 *fit_one_cycle* 方法支持的所有参数:*

*   *[https://docs.fast.ai/train.html#fit_one_cycle](https://docs.fast.ai/train.html#fit_one_cycle)*

# *我们取得了什么成就？*

*使用 448x448 的图像分辨率和 fast.ai，我们在实验中使用的三个网络模型中的两个(ResNet34 和 ResNet50)获得了大约 93%的验证准确性。这比上面教程的 85%要好得多，上面的教程使用了 MobileNet，图像分辨率为 224x224，Keras。[(目前)ka ggle(employees tensor flow . js)上投票最多的内核获得了 86%的精度](https://www.kaggle.com/vbookshelf/skin-lesion-analyzer-tensorflow-js-web-app)。*

*在任务#3 的 ISIC 2018 挑战赛最终测试中，**排名最高的竞争对手**，来自【MetaOptima Technology Inc .【的 Jordan Yap 取得了 95.8% 的准确率和由 ISIC 选择的竞争对手评估标准的*平衡多类准确率 88.5%。Jordan Yap 采用的方法基于:**

*   *附加的外部数据[33，644 张图像]；*
*   *相当于原始 ImageNet 分辨率的低分辨率图像；*
*   *19 种分类算法的集合，其中一种不是神经网络(直方图分析)；*
*   *一个 XGBoost 分类器，它在这个集成的结果之上被训练。*

*![](img/77a810c0bf835e4bf978a41846cd80b0.png)*

*Aleksey Nozdryn-Plotnicki, Jordan Yap, and William Yolland, MICCAI, 2018*

*[Aleksey Nozdryn-plot nicki、Jordan Yap 和 William Yolland 提交给 *ISIC 皮肤图像分析研讨会和 Challenge @ MICCAI 2018* 的论文及其结果在此。](https://s3.amazonaws.com/covalic-prod-assetstore/c9/21/c921175d59914fa79e8098c4ead60cfa?response-content-disposition=inline%3B%20filename%3D%22ISBI_2018.pdf%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires=3600&X-Amz-Credential=AKIAITHBL3CJMECU3C4A%2F20190705%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-SignedHeaders=host&X-Amz-Date=20190705T183010Z&X-Amz-Signature=7cbc21be5a9e515a83b44b48608b0c76436eff104c2d60a49e379ed4636dcf6e)*

*我们的结果与最初的 ISIC 2018 挑战赛没有直接可比性，因为 ISIC 提供了从 HAM1000 数据集手动提取的 1512 幅图像的测试集，所有竞争对手都必须采用该测试集并将其结果提交给 ISIC。我们使用从 HAM1000 数据集中随机提取的一组 2003 张图像来验证我们的训练，并使用剩余的 8012 张图像来训练我们的网络。*

*![](img/f34ecc647453750797eb9e0adbd98401.png)*

*Statistics of Jordan Yap from MetaOptima Technology Inc.*

*您可以在此查看 ISIC 2018 结果:*

*   *[ISIC 2018 —任务 3:最终测试](https://challenge.kitware.com/#phase/5b1c1aa756357d41064da300)*
*   *[ISIC Challenge 2018 排行榜](https://challenge2018.isic-archive.com/leaderboards/)(点击“任务 3”窗格)*

*然而，有趣的是，我们在 NVIDIA T4 GPU (4.2 x 10 + 4.3 x 35 分钟)上使用 fast.ai 和一个网络 ResNet34 以及总训练时间为 190 分钟的**获得的 **92.9%** 的准确性，仅比 ISIC 2018 挑战赛最高排名的竞争对手使用更复杂的方法获得的准确性差 **2.9%*****

*这很可能是因为我们采用了两倍的图像分辨率，允许更多的细节，但 fast.ai 中的 HYPOs(超参数优化)可能也发挥了作用。*

# *我们学到了什么？*

***ResNet34 入手**是个不错的选择:ResNet 在 fast.ai 中确实管理的很好，有各种好用的 HYPOs(超参数优化)。如果你不知道使用哪种网络，可以使用 ResNet34，它足够小，即使你在家里使用 GPU 也可以相对快速地进行训练，但也足够大，可以表示一大组问题。ResNet34 trains 将如何提供提示，如果你应该与你的网络规模上升或下降。*

***盲目接受学习率建议并不总是最佳选择**:resnet 50 的 *lr_find()* 产生了一个很长的平稳期，该方法在图形左端的一个小山谷中建议了一个非常低的学习率值。当我们用这个值训练网络时，它会振荡，没有产生好的结果(只有 90%的准确率)。当我们对图表进行可视化分析，并将一个高一个数量级的值作为学习率范围的下限时，该值位于平台较平坦部分的开始，ResNet50 学习得更好，并达到与 ResNet34 相同的 93%的准确率。所以，使用*建议=真*模式，但是在你接受它之前给它一个严肃的处理，*实际上看着*图形。这是学习率范围的第二条经验法则:查看整个图形并找到真正的平稳段——学习率范围的理想下限将位于该平稳段的开始处。*

*方法[*learn . lr _ find()*](https://docs.fast.ai/callbacks.lr_finder.html)帮助你找到一个最优的学习率。它使用了 2015 年论文*中开发的技术，用于训练神经网络*([http://arxiv.org/abs/1506.01186](http://arxiv.org/abs/1506.01186))的循环学习率，我们只是从一个非常小的值开始增加学习率，直到损失开始减少。如果你想知道更多关于寻找最佳学习率的信息，请看这里:*

*   *[**迈向数据科学**:*用 LR-Finder 加速神经网络训练——为你的网络找到良好的初始学习速率*作者 Faizan Ahemad](/speeding-up-neural-net-training-with-lr-finder-c3b401a116d0)*

***越大并不总是越好**:最终，ResNet50 的表现几乎与 ResNet34 相同，但训练时间要长得多，结果也稍差。用大型网络模型开始您的培训空间探索是一个糟糕的选择。*

***图像分辨率发挥作用**:我们采用了 ISIC 2018 挑战赛最高排名竞争对手的两倍分辨率，通过一个相对简单的 ResNet34 获得了可比的结果，而该竞争对手在 18 个不同的网络上采用了机器学习方法，包括一个巨大的 ResNet152。*

***Fast.ai 很快:**最后，与其他方法相比，使用 *fast.ai* 我们能够使用更少的代码解决相同的分类问题，同时使用高级超参数优化策略，使我们的训练速度更快。同时，一组高级功能也允许我们以表格和图形的形式轻松检查结果。这种简单性使我们可以用三种不同的网络模型进行实验，并比较它们的结果。这表明 fast.ai 是更传统的 CNN 框架的一个非常有前途的替代方案，特别是如果手头的任务是图像分类、对象检测或语义分割等“标准”深度学习任务，可以通过微调现成的预训练网络模型来解决。*

# *承认*

*这项工作是除我之外的一组积极参与的研究人员共同努力的结果:*

*   *丹尼尔·霍尔特豪森努内斯<[daniel@floripa.com.br](mailto:daniel@floripa.com.br)*
*   *路易斯·安东尼奥·布舍托·马卡里尼<*
*   *马塞洛里卡多斯特梅尔<[马塞洛. stemmer @ ufsc . br](mailto:marcelo.stemmer@ufsc.br)*

*我们还要感谢于尔根·克鲁施<[、juergen.kreusch@gmail.com](mailto:juergen.kreusch@gmail.com)>和路易斯·费尔南多·科普克< [、luiskopke@uol.com.br](mailto:luiskopke@uol.com.br)>提供的立体皮肤镜材料。*

# *参考*

*[1]Kreusch，J. [*入射光显微术:对活体皮肤显微术的思考*](https://www.ncbi.nlm.nih.gov/pubmed/1459756#) 。国际皮肤病学杂志。 1992 年 9 月；31(9):618–20.*

*[2]Kopke，L.F. [*皮肤镜在基底细胞癌早期发现、控制和手术规划中的作用*](http://www.surgicalcosmetic.org.br/exportar-pdf/3/3_n2_126_en/A-dermatoscopia-na-deteccao-precoce--controle-e-planejamento-cirurgico-dos-carcinomas-basocelulares) 。2011 年皮肤美容外科手术；3(2):103–8.*