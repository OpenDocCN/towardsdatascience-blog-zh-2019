# 用 C++部署机器学习模型

> 原文：<https://towardsdatascience.com/machine-learning-model-deployment-with-c-fad31d5fe04?source=collection_archive---------6----------------------->

最近，我着迷于构建一个数学化的应用程序，并在没有任何模型大小、平台或 api 调用需求限制的情况下大规模部署它，这将是多么有趣。我知道 Python 有足够的库来处理机器学习项目的原型，但是没有多少人在谈论扩展这个项目，尤其是当你不想通过 web api 来做这件事的时候。

我相信真正的智能不应该仅仅依赖于对 api 的调用来实现模型的规模化。这种魅力让我开始研究如何将 C++用于机器学习和一般智能。

我坚信 Matlab 和 Python 的数学优势都是基于底层的 c/c++代码。因此，从根本上扩展技术以处理机器学习中涉及的数学计算，并考虑一个极快的场景，可能需要您能够深入研究低级编程，尤其是 c/c++。

此外，我想知道为什么大多数计算机科学学校确保他们的课程中有 c/c++课程。这强调了将 c/c++用于可伸缩技术的理由。

在使用 Udemy 动手课程学习 c++之后，现在的挑战是在 android 中集成一个简单的人脸识别应用程序。

这篇文章将包括构建 c++项目和在 android 或任何其他操作系统环境中部署所需的一些初步方法。

**组件**:

1.  用 opencv 建立一个机器学习的 c++项目。
2.  学习主成分分析生成特征脸
3.  为多平台部署设置. so 推理库
4.  为推理库开发 jni 包装器。
5.  使用 Android 或任何 java 代码中的库。

第一部分将是用 OpenCV 学习一个机器学习算法。在这种情况下，我们将探索最基本的人脸识别算法，使用主成分分析的特征脸。机器学习社区非常熟悉 python 中的这一点，特别是使用 sci-kit learn 等工具，但是当生产，尤其是离线/设备上生产出现在脑海中时，从不同的维度进行这一工作的需求是有利的。

OpenCV 提供了一个非常好的学习主成分分析的 api，一旦设置好数据，学习起来非常简单。

# **以下是步骤:**

## 使用 OpenCV 建立一个 cmake 项目 c++。

您的 CMakeFile.txt 的关键部分是确保 0 项目中有一个 OpenCV 库，并且可供您的库编译。理想情况下，在请求 Cmake 为您查找 OpenCV 之前，在您的机器上安装 OpenCV 库是很重要的。

```
find_package(OpenCV REQUIRED)include_directories(${OpenCV_INCLUDE_DIRS})set(LIBS ${OpenCV_LIBS})target_link_libraries(featureCalculation ${LIBS})
```

PS:为了训练和推断，我不得不设置不同的 cmake。我将分享这两个

## **学习主成分分析。**

既然 OpenCV 已经可用，学习 PCA 就相当简单了。逻辑是这样的:

***读取所有图像数据作为数组。*** 使用 OpenCV 中的 ***cv::glob*** ，所有文件名都以 ***结尾。jpg，。png 或/和。用 ***cv::imread*** 可以读取 jpeg*** ，可以进行图像数据的数据预处理。

***农作物面孔。*** 这很重要，因为PCA 对人脸图像比对整幅图像做得好得多。我发现*多任务级联卷积网络(MTCNN)* 是最可靠、最简单、最小的人脸检测和裁剪模型。在 C++中有一个使用 OpenCV 中的 Caffe 网络的原始模型的实现。

***将裁剪后的面转换成灰度。*** 这部分挺直截了当的。使用***cv::CVT COLOR(originalimagemat，grayscaleimagematcontainer，cv::COLOR_BGR2GRAY)*** 我们可以在 OpenCV 中将原始的 BGR 图像转换成灰度。

***其他预处理*** —另一个预处理是确保数据类型正确。这非常重要，因为 C++非常注重数据类型的精度。在这一点上很容易引入错误，因此要小心确保数据类型是正确的。除此之外，这是一个好主意，规范化您的图像数据和调整所有图像到一个一致的形状。只有当数据在同一维度时，PCA 才起作用。在 OpenCV 中，我们可以使用以下函数来处理预处理:

***CV::resize(original image，containermatofnewimage，size)*** 用于调整图像大小***original mat::convertTo(newmatnormalize，CV_32FC3，1/255.0)*** 用于归一化图像的像素值。

***将所有图像转换成数据表*** —数据表有点像单个数据表，其中每个元素都表示为一行，有趣的是，我们可以将数据表中的每一行都视为扁平格式的独立图像。PCA 的本质是将图像值投影到具有该图像的独特表示的几列。因此，数据表将具有等于训练数据集中图像数量的行，而列将是每个图像的归一化灰度值。

为了创建数据表， ***std::vector*** 可用于保存所有图像(希望它们适合内存)，然后将其复制到数据矩阵的每一行。这是一个帮助器函数，它从一个矢量图像 mat 中完成这个任务。

```
**cv::Mat** **createdatamatrix**(std::vector<cv::Mat> imageArray) {
cv::Mat datamatrix(static_cast<int>(imageArray.size()), imageArray[0].rows * imageArray[0].cols, CV_32F);
unsigned int i;
*for*(i=0; i < imageArray.size(); i++) {
   cv::Mat imageRow = imageArray[i].clone().reshape(1, 1);
   cv::Mat rowIData = datamatrix.row(i);
   imageRow.copyTo(rowIData);
  }
  ***return*** datamatrix;
}
```

***cv::reshape()*** 帮助将 mat 数组变换成不同的形状，用 ***(1，1)*** 字面意思是我们希望数据存在一行中。

***学习实际主成分分析算法。*** 现在我们已经用预处理的人脸图像创建了数据表，学习 PCA 模型通常是顺利的。就像把数据表传递给一个 OpenCV pca 实例一样流畅，带有你期望的最大组件像这样***cv::PCA PCA PCA(datatable，cv::Mat()，cv::PCA::DATA_AS_ROW，number_of_components)*** 。这样，我们就有了一个用 C++编写的学习过的 PCA，可以用于生产了。

为了将该模型转移到任何环境中使用，open cv 有一个 ***文件存储*** 对象，允许您按原样保存垫子。因此，我可以保存这个文件，并通过 jni for OpenCV 传递其文件名，以重新创建用于推理的模型实例。为模特服务就是这么简单。

为了结束本文的推理部分，我将简单地展示如何使用 OpenCV 编写 mat 对象。最终，保存的模型中的值会以 YAML 文件或 XML 的形式输出，这取决于用户最喜欢的选择。

***保存 pca 模型对象，用于生产环境上的推理。*** pca 对象中确切需要保存的是经过训练的 PCA 的*均值*和*特征向量*，有时也保存*特征值*可能是个好主意，以防您想要构造自己的特征脸投影，但是 OpenCV 已经实现了一个 ***pca- >项目*** 实例，它有助于推理和特征脸生成。在任何情况下，以下是如何保存您的模型:

```
**void Facepca::savemodel**(cv::PCA pcaModel, const std::stringfilename)
{
  cv::FileStorage fs(filename,cv::FileStorage::WRITE);
  fs << “mean” << pcaModel.mean;
  fs << “e_vectors” << pcaModel.eigenvectors;
  fs << “e_values” << pcaModel.eigenvalues;
  fs.release();
}
```

## 推理层

保存模型后，推理层将包括加载保存的模型，该模型的值也通过 OpenCV 存储在 yml 文件中，并被开发以形成具有必要数据预处理模块的推理引擎，该推理引擎最终被捆绑为用于在基于 linux 的环境上部署的. ***so*** 文件。对于其他平台，cpp 代码可以被编译为分别用于 windows 和 mac 的 ***dll*** 或 ***dylib*** 。

因为我们更关心在 android 应用程序上部署模型，所以重点将是构建 ***。所以*** 文件推理机从我们保存的模型中。

## 建筑。so 推理库

***加载 PCA 模型*** 为了推理，我们让 OpenCV 从保存的 ***中加载已有的模型文件。yml*** 文件，之后我们将特征值、特征向量和均值输入到一个新的 ***PCA*** 对象中，然后我们可以调用一个 ***pca- >项目*** 来创建一个新图像的投影。

下面是加载保存的 OpenCV 文件存储模型的示例代码。

```
cv::PCA **newPcaModel**;
cv::PCA **loadmodel**(cv::PCA newPcaModel, const std::string filename){
   cv::FileStorage fs(filename,cv::FileStorage::READ);
   fs[“mean”] >> newPcaModel.mean ;
   fs[“e_vectors”] >> newPcaModel.eigenvectors ;
   fs[“e_values”] >> newPcaModel.eigenvalues ;
   fs.release();
   ***return*** newPcaModel;
}
**loadmodel**(newPcaModel, “path-to-saved-yml-file”);
```

一旦模型被加载， ***newPcaModel*** 对象现在包含从现有训练参数保存的模型，即 pca 特征值、特征向量和均值。因此，当完成面部投影时，可以保证返回的数据与训练数据集相关。

***创建新的图像预处理和预测阶段。*** 在机器学习模型的推断过程中，重要的是输入图像也要经过与训练数据集相同的预处理。

可以使用几种方法将图像传递给推理引擎，可能是从磁盘加载图像，也可能是将图像作为 base64 字符串传递。

在我们的例子中，可能的方法是使用 base64 字符串，因为我们也考虑了两个因素，一个是 ***jni*** 将被暴露，另一个是我们的最终库是用于 android 应用程序的。

记住这一点，然后我们需要确保库能够从 base64 字符串中检索图像并将其发送到 OpenCV。

在 c++中解码一个 base64 字符串是非常重要的，然而，读者可以参考[这个链接](https://renenyffenegger.ch/notes/development/Base64/Encoding-and-decoding-base-64-with-cpp)中的代码片段。

一旦 base64 图像字符串被解码，我们就将该字符串转换成一个由 *无符号字符(uchar)* 组成的*向量，它可以被认为是图像值。*

OpenCV 可以使用函数调用***cv::im decode(vectorUchar，flag)*** 将 uchar 的矢量解码成图像。这个过程返回一个 ***Mat*** 图像，用它可以做进一步的预处理。

然后，图像可以通过以下预处理阶段:

*   人脸提取
*   将裁剪面转换为灰度。
*   图像大小调整
*   图像标准化
*   创建数据矩阵

正如本文上面的培训部分所述。

对新图像进行推理的最后一步是使用*加载的* pca 对象在新图像中创建人脸的投影。

这样做的代码片段将如下所示:

```
newPcaModel->project(datamatrix.row(0));
```

当你用一个*加载的* pca 对象在两个面上投影，并使用距离度量(如欧几里德距离或余弦相似度)比较投影(特征面)时，识别或验证部分发生。

## 为推理库开发 jni 包装器

***用一个暴露的 jni 打包这个库。*** 这一部分非常简单，一旦推理代码被适当地结构化，很可能是在一个类中，那么 jni 主体就可以直接调用暴露的函数来进行模型加载、预处理和预测。

然而，要创建一个 jni，理解 java 中的链接是如何发生的是很重要的。

第一种方法是创建一个 java 类和您希望在 Java 类中使用的带有输入参数的函数。

这将是 jni 在创建 cpp 代码时使用的函数名。Java 中方法的类路径名与 cpp 中的类路径名之间的一致性非常重要。

让我们假设我们将用于 pca 特征匹配的方法被命名为***matchpcafeatures()***，然后我们的 java 类可以看起来像这样。

```
package org.example.codeclass MatchFeatures {static float matchpcafeatures(modelfilename: String, image: String, projectionToCompare: Array[Float])}
```

有了上面的 java 类和方法，我们的 jni 头文件看起来会像这样。

```
*#include* <jni.h>*/* Header for class com_example_code_MatchFeatures */
#ifndef* _Included_com_example_code_MatchFeatures
*#define* _Included_com_example_code_MatchFeatures*#ifdef* __cplusplus
extern “C” {
*#endif*JNIEXPORT jfloat JNICALL Java_com_example_code_MatchFeatures_matchpcafeatures
(JNIEnv *, jobject, jstring, jstring, jfloatArray);*#ifdef* __cplusplus
}
*#endif
#endif*
```

你还不需要担心 ***extern C*** 的细节，重点是 jni 头文件中方法的名称。

接下来，我们将看看如何使用上面的 java 代码进行推理。让我们首先为这个方法开发 jni 桥。

jni 头的最后 3 个参数与 java 方法中的参数完全相同，位置也完全相同。

因此，我们将对这些参数进行处理，因为它们是客户对我们的 c++推理引擎的关键要求。

下面展示了如何连接这些输入参数并返回一个值，java 代码可以接受这个值并继续它的其他进程。

```
*/**
* Match features of pca
* */*JNIEXPORT jfloat JNICALL Java_com_seamfix_qrcode_FaceFeatures_matchpcafeatures
(JNIEnv * env, jobject obj, jstring pcafilename, jstring imagestring, jfloatArray projectionToCompare){ const char *pcastring_char;
  pcastring_char = env->GetStringUTFChars(pcafilename, 0);
  *if*(pcastring_char == NULL) {
    *return* 0.0f;
  } const char *imagestring_char;
  imagestring_char = env->GetStringUTFChars(imagestring, 0);
  *if*(imagestring_char == NULL) {
     *return* 0.0f;
  } *//Get file name string as a string for cpp* std::string stdfilename(pcastring_char);
  cv::PCA pca; *//Class InferencPca holds the preprocesing and inference methods* InferencePca ef;
  std::vector<cv::Mat> imagevec; *//Get image as base64* cv::Mat image = ef.readBase64Image(imagestring_char);
  ef.loadmodel(pca, stdfilename);
  imagevec.push_back(image); cv::Mat datamatrix = ef.createdatamatrix(imagevec);
  cv::Mat projection = ef.project(datamatrix.row(0)); *//Load the existing vector.* std::vector<float> initialProjectionPoints; *//Load existing features -- needed to do comparison of faces* jsize intArrayLen = env->GetArrayLength(existingfeatures);
  jfloat *pointvecBody = 
      env->GetFloatArrayElements(existingfeatures, 0); *for* (int i =0; i < intArrayLen; i++) {
     initialProjectionPoints.push_back(pointvecBody[i]);
  } std::vector<float> newProjectionpoints =     
                ef.matToVector(projection);
  float comparisonScores = 
     ef.compareProjections(newProjectionpoints, 
                            initialProjectionPoints); env->ReleaseFloatArrayElements(existingfeatures, pointvecBody, 0);
  env->ReleaseStringUTFChars(pcafilename, pcastring_char);***return*** comparisonScores;}
```

有了这个，我们就着手打造我们的 ***。于是*库和**库成功地释放给 java 代码使用。***cmakelists . txt***中说明了构建库的协议。

它看起来像下面这样:

```
find_package(JNI REQUIRED)*#Include jni directories* include_directories(${JNI_INCLUDE_DIRS})
file (GLOB_RECURSE SOURCE_FILE src/*.h src/*.cpp)set(LIBS ${JNI_LIBRARIES})
add_library(matchprojection SHARED ${SOURCE_FILE})target_link_libraries(matchprojection ${LIBS})
```

构建项目应该生成一个***lib-match projection . so***文件，该文件可以添加到您的 java 项目中。

然而，对于 android 来说，这有点棘手，因为构建工具不同于官方的 cmake 构建工具链，Android 有自己的本地 cpp 代码构建工具。这叫做***【NDK】***【原生开发套件】。这将用于为生成的 ***构建 c++原生代码。所以*** 要兼容安卓。

**建筑*。所以使用 NDK 的安卓系统的*** 将会是一个完整的教程，所以我现在跳过它。

但是一般来说，一旦使用 NDK 完成构建，您将拥有相同的***lib-match projection . so***可以在您的 android 应用程序中使用。

## 使用。Android 应用程序中的库。

在 Android 应用程序中使用生成的库就像在任何 java 应用程序中使用它一样。

其思想是加载本地库，然后用所需的参数调用最初创建的类中对应于本地 jni 方法的方法。

要在包括 android 在内的任何 java 程序中加载该库，请确保 ***。所以*** 库是在你的程序的类路径中，有的会把它放在一个名为 ***lib*** *或者* ***jniLibs*** 的文件夹下。这样我就可以使用函数调用来加载库，如下所示:

```
System.loads(“native-lib”)
```

最后，我可以用必要的参数调用前面创建的方法，然后本机代码可以为我执行。

```
MatchFeatures mf = *new* MatchFeatures();float matchscores = mf.matchpcafeatures(storedpacfilepath, imagebase64string, anotherimageprojectionarray);
```

如果您仔细观察，就会发现该方法被声明为 native，并且该方法没有主体，这是因为程序知道有一个用该类路径名定义的 native cpp 方法。

# 结论:

这种方法是在 java 环境中构建和部署包含本机代码的项目的基本方法。此外，最复杂的算法问题，包括机器学习和核心计算机视觉项目，可以很容易地在 cpp most 中推理，因为它很快，而且有现成的库。

甚至 TensorFlow 也有一个在 c++中加载深度学习模型以及在 c++中使用其 ***tflite*** 模型的 api。

因此，我认为这种使用本机代码构建推理引擎的方法是一种构建强大的生产就绪型引擎的方法，这种引擎将利用高精度数学，尤其是将机器学习模型部署到各种环境，尤其是离线环境中的 android 环境。

*我在 Seamfix Nigeria Ltd .担任数据科学家和机器学习工程师，我的工作重点是确保数据科学功能最终应用于生产。*