# 使用 Docker 检测文本中的希腊语，并在没有 Boto3 的情况下存储在 S3。

> 原文：<https://towardsdatascience.com/detect-greek-language-in-text-using-docker-store-in-s3-without-boto3-3c6d6b220639?source=collection_archive---------34----------------------->

![](img/bcc70de51d7df81c30e8b87d9f4c13dc.png)

***嗨，你好！在这篇文章中，我们将构建一个小的“应用程序”，它能够检测文本中的希腊语并在 AWS S3 中存储一些结果。***

更详细地说，该应用程序将是一个 docker 容器，它将接受输入文件路径，将能够获取文件(例如，从 AWS S3 桶)，使用 python 脚本来处理它，检测文本中的希腊语，仅保留希腊语文本，并最终上传结果。希腊文本的 csv 文件放回 AWS S3 桶。

作为第一步，我们将构建一个 Docker 映像&确保语言检测、文本处理和 AWS 通信所需的所有模块都安装在映像的虚拟 OS 中，只要保存 AWS 凭证的环境变量被正确地传递到 Docker 映像中。

更有趣的是:

*   我们将不使用 boto3 库，因为当 AWS S3 的访问权限严格时，它并不总是与 AWS CLI 功能相同。
*   我们将使用 [fastText](https://fasttext.cc/) 文本分类库，它实际上将是我们的语言检测工具，因为它提供了微小的 [lid.176.ftz](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz) 模型；其对应主模型的压缩版本，文件大小仅为 917kB！

# 需要的文件

在设置我们的流程之前，我们需要准备一些文件:

*   。:在目录中创建一个本地文件夹，docker 文件将放在其中并用于构建映像。
*   。/init/:在上面的主文件夹中，创建一个' init '文件夹，该文件夹将包含所有需要复制到 docker 映像中的文件。
*   。/init/envs.py:环境变量文件:

```
envs = {
    "LOCAL_FILE": 'temp.csv',
    "LOCAL_OUT_FILE": 'output.csv',
    "AWS_S3_PROFILE": 'my_aws_s3_profile_name'
}
```

*   。/init/s3.py:主要功能的脚本。
*   。/init/lib _ process . py:S3 . py 文件导入和使用的帮助器库。
*   。/init/lid.176.ftz:语言检测模型。
*   。/init/credentials:AWS 凭据文件:

```
[my_aws_s3_profile_name]
aws_access_key_id = ABCDE
aws_secret_access_key = 12345
```

*   。/init/config:AWS 配置文件:

```
[my_aws_s3_profile_name]
output=json
region=us-east-2
```

其中我们假设我们的 AWS 区域是 us-east-2。

*   Dockerfile:用于构建 docker 映像的 docker 指令。

# **Docker 图像**

让我们一起逐行解析**docker 文件**:

```
1 FROM python:3.6.6-slim
2 WORKDIR .
3 COPY ./init/envs.py .
4 COPY ./init/s3.py .
5 COPY ./init/lib_process.py .
6 COPY ./init/lid.176.ftz .
7 RUN mkdir -p /.aws/
8 RUN apt-get update && apt-get install -y \
9         python-dev \
10        python-numpy \
11        python-scipy \
12        && rm -rf /var/cache/apk/*
13 COPY ./init/credentials /.aws/
14 COPY ./init/config /.aws/
15 ENV AWS_CONFIG_FILE=/.aws/config
16 ENV AWS_SHARED_CREDENTIALS_FILE=/.aws/credentials
17 RUN pip3 install --upgrade awscli
18 RUN pip3 install fasttext
19 RUN chmod u+x s3.py
20 ENTRYPOINT ["python3","s3.py"]
21 CMD ['']
```

*按大小构建最佳 docker 图像超出了范围；在《走向数据科学》中有许多关于这方面的好文章:-)*

*   **第一行:**我们从一个 python3.6 小图开始。
*   **第 2 行:**我们将当前目录设置为构建映像时看到的目录。
*   **第 3 行:**将 env 移动到映像的操作系统中。
*   第 4 行& 5: 用 python 脚本做同样的事情。
*   **第 6 行:**对语言检测模型做同样的操作。
*   第 7 行:创建 AWS-CLI 设置目录。
*   **第 8–12 行:**通过 docker 镜像的 Ubuntu OS 安装一些基本的 python 模块，但是不保留它们的安装文件。
*   **第 13 行& 14:** 将 AWS 设置文件(包括 AWS-CLI 凭证)移动到映像的操作系统默认 AWS-CLI 设置目录中。✧:这很重要。我们在构建时注入凭证，而不是在 docker 文件或相关源代码(共享代码)中硬编码凭证。
    ✧ *让运行中的容器要求凭证作为输入参数会更安全。* ✧ *以上是我们通过 envs.py 文件传递凭证的例子。*
*   **第 15 行& 16:** 刷新(设置)显示设置文件路径的 AWS 环境变量。
*   **第 17 行:**通过 python 的 pip 安装 AWS-CLI。
*   **第 18 行:**通过 python 的 pip 安装 fasttext 库。
*   **第 19 行:**使 s3.py 文件可执行。
*   **第 20 行& 21:** 默认情况下，上面的映像创建的任何容器都将使用虚拟操作系统的 python3 来运行 python 可执行文件 s3.py。

# 使用 docker 图像

我们可以运行以下命令，例如在 Ubuntu 中:

*   建立形象:

```
docker build --no-cache -t <image_name> .
```

*   创建并运行图像的容器:

```
docker run --rm -t --name <container_name> <image_name> \
"s3://<bucket_name>/<relative path to file>/input.csv" \ "s3://<bucket_name>/<relative path to file>/output.csv"
```

在我们的例子中，上面的意思是我们创建并运行一个容器，它将在第一个 s3 路径中找到输入文件，并将处理后的结果文件存储在第二个 s3 路径中。

*   通过打开虚拟操作系统进行手动探索来调试映像:

```
docker run --entrypoint "/bin/bash" -it \ 
--name <container_name> <image_name>
```

# **内部流动**

正如我们在上面看到的，容器使用一些给定的参数执行 s3.py 脚本。剧本:

*   读取参数。
*   通过其主机操作系统(即 docker 映像操作系统)运行 aws-cli 命令，并从 aws s3 获取输入。csv 文件。
*   处理输入文件并生成输出文件。
*   再次运行 aws-cli 命令，并将其上传到 s3 输出中。csv 文件。
*   对于本教程。csv 文件是 utf-8 编码的，用逗号分隔。

**s3.py** 脚本:

```
#!/usr/bin/python3import os
import sys
from envs import envs
from lib_process import processingif __name__ == '__main__':
    print(os.uname())
    print("This is the name of the script: ", sys.argv[0])
    print("Number of arguments: ", len(sys.argv))
    print("The arguments are: ", str(sys.argv))
    REMOTE_IN_FILE = sys.argv[1]
    LOCAL_FILE = "./" + envs["LOCAL_FILE"]
    LOCAL_OUT_FILE = "./" + envs["LOCAL_OUT_FILE"]
    REMOTE_OUT_FILE = sys.argv[2]
    print("Entered the python script.")
    print("Envs: REMOTE_IN_FILE:", REMOTE_IN_FILE)
    print("Envs: LOCAL_FILE:", LOCAL_FILE)
    print("Envs: REMOTE_OUT_FILE:", REMOTE_OUT_FILE)
    print("Envs: AWS_S3_PROFILE:", envs['AWS_S3_PROFILE']) down_command = "aws s3 cp " + "--profile " + 
        envs['AWS_S3_PROFILE'] + " " + REMOTE_IN_FILE + " " 
        + LOCAL_FILE
    print(down_command)
    os.system(down_command) processing(LOCAL_FILE, LOCAL_OUT_FILE) up_command = "aws s3 cp " + "--profile " + 
        envs['AWS_S3_PROFILE'] + " " + LOCAL_OUT_FILE + " " 
        + REMOTE_OUT_FILE
    print(up_command)
    os.system(up_command)
```

# 处理

s3.py 脚本使用包含执行处理的函数的助手库 python 文件。

**假设输入文件的行是这样的格式:"< text_id >，< text > \n"**

**lib_process.py** 库:

```
import fasttext def is_greek(txt, model):
    # Predict only the most dominant language in the text:
    pred = model.predict(txt, k=1)
    # Fetch the found language
    pred_langs = pred[0]
    pred_lang = pred_langs[0]
    # Compare the predicted language with the greek lang label:        
    return False if pred_lang != '__label__el' else True def processing(local_file, out_file):
    # Open the file readers:        
    fin = open(local_file, mode='r')
    fout = open(out_file, mode='a') # Load the language detection model, once:
    model = fasttext.load_model('lid.176.ftz') line_count = 0    
    while True: # Loop over all input lines
        # Split text by sentence
        line = fin.readline()
        if not line:
            break # File ended
        line_count += 1 # Split line(sentence) into tokens & keep the text
        txt = line.strip().split(",")[1]

        # Keep Greek sentences
        if is_greek(txt, model):
            # Store in the output file the found greek sentence:            
            fout.write(line + '\n') # Close the files:
    fin.close()
    fout.close()
```

# **原来如此！我希望你喜欢它！**

感谢阅读，请不要犹豫留下任何评论或分享意见。