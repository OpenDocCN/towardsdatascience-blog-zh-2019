# 用这段 Python 代码自动编辑你的视频！

> 原文：<https://towardsdatascience.com/edit-your-videos-automatically-with-this-python-code-c704cc413a1?source=collection_archive---------13----------------------->

## 是的，这是真的，小心编辑！！谢谢 Python。

时候到了！我正在为商业科学和 T2 科学等令人惊叹的公司制作课程，我正在编辑视频，这是一件痛苦的事情！！我在这里给你展示的软件是由伟大的[卡里克](https://www.youtube.com/user/carykh)开发的。

该系统的目标是编辑视频中令人讨厌的沉默，我们犯的错误，以及几乎所有你喜欢的东西。

它不会把你的视频变成一个美丽的视频，但至少它会让你省下一小时又一小时听自己失败的时间。

如果你不是视频制作者，但你在网上观看视频，这可以帮助你更快地观看。我记得当我在学习的时候，我看了很多斯坦福和麻省理工的视频，花了我几个小时，很多时候教授甚至不说话。所以这个也能帮上忙。

# 装置

我在 MatrixDS 上测试了这个系统，matrix ds 是在几秒钟内免费推出终端和 Jupyter 笔记本的最佳工具之一。所以你可以在那里复制它。

您需要做的第一件事是克隆回购:

[](https://github.com/carykh/jumpcutter.git) [## 卡里克/jumpcutter

### 自动编辑 vidx。这里解释一下:https://www.youtube.com/watch?v=DQ8orIurGxw-卡里克/jumpcutter

github.com](https://github.com/carykh/jumpcutter.git) 

```
git clone [https://github.com/carykh/jumpcutter.git](https://github.com/carykh/jumpcutter.git)
```

然后安装要求:

```
cd jumpcutter
pip install --user -r requirements.txt
```

你还需要 ffmpeg。如果你在 Ubuntu 上:

```
sudo apt update
sudo apt install ffmpeg
```

如果您在 MatrixDS 中，您需要成为 root 用户，因此:

```
sudo su
apt install ffmpeg
```

# 使用

现在是有趣的部分！我制作了一个简单的视频来测试这一点。这里可以看到原视频:

我在 MatrixDS 里做的是把视频上传到克隆的 repo 的同一个文件夹里。然后在运行该命令后:

```
python3 jumpcutter.py --input_file auto_2.mp4 --sounded_speed 1 --silent_speed 999999 --frame_margin 2
```

我在几秒钟内就编辑好了我的视频。秒！！

您可以在这里看到结果:

这简直太棒了:)

现在还没有太多关于这个软件的文档，但是代码是开源的，所以你可以去回购看看所有的东西。

基本命令是

```
python jumpcutter.py --input_file path/to/file.mp4
```

您可以使用以下选项:

```
optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        the video file you want modified
  --url URL             A youtube url to download and process
  --output_file OUTPUT_FILE
                        the output file. (optional. if not included, it'll just modify the input file name)
  --silent_threshold SILENT_THRESHOLD
                        the volume amount that frames' audio needs to surpass to be consider "sounded". It ranges from 0 (silence) to 1 (max
                        volume)
  --sounded_speed SOUNDED_SPEED
                        the speed that sounded (spoken) frames should be played at. Typically 1.
  --silent_speed SILENT_SPEED
                        the speed that silent frames should be played at. 999999 for jumpcutting.
  --frame_margin FRAME_MARGIN
                        some silent frames adjacent to sounded frames are included to provide context. How many frames on either the side of speech
                        should be included? That's this variable.
  --sample_rate SAMPLE_RATE
                        sample rate of the input and output videos
  --frame_rate FRAME_RATE
                        frame rate of the input and output videos. optional... I try to find it out myself, but it doesn't always work.
  --frame_quality FRAME_QUALITY
                        quality of frames to be extracted from input video. 1 is highest, 31 is lowest, 3 is the default.
```

是啊！你甚至可以通过一个 YouTube 视频链接，它会下载原来的视频，并改变视频创建一个新的。为此，该命令将如下所示:

```
python3 jumpcutter.py --url [https://youtu.be/2MjlMpEzDA8](https://youtu.be/2MjlMpEzDA8) --sounded_speed 1 --silent_speed 999999 --frame_margin 2 --frame_rate 3
```

仅此而已。

可爱的作品。顺便说一下，这些代码只有 203 行代码！这些天来，Python 的力量让我感到惊讶。

感谢你阅读这篇文章。如果您有任何问题，请在此写信给我:

[](https://www.linkedin.com/in/faviovazquez/) [## 法维奥·瓦兹奎——science y Datos | LinkedIn 创始人/首席数据科学家

### 加入 LinkedIn ‼️‼️重要提示:由于 LinkedIn 技术限制，我现在只能接受连接请求…

www.linkedin.com](https://www.linkedin.com/in/faviovazquez/) 

祝学习愉快:)

![](img/e42964ce99613fd55fbcb27ea45dfa30.png)