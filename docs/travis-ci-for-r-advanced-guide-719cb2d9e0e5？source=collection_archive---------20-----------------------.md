# Travis CI for R —高级指南

> 原文：<https://towardsdatascience.com/travis-ci-for-r-advanced-guide-719cb2d9e0e5?source=collection_archive---------20----------------------->

## 在 Travis CI 中构建 R 项目的持续集成，包括代码覆盖、`[pkgdown](https://github.com/r-lib/pkgdown)` [](https://github.com/r-lib/pkgdown)文档、osx 和多个 R 版本

![](img/7aadd0c52a29b58ac29766401c9d3164.png)

Photo by [Guilherme Cunha](https://unsplash.com/@guiccunha?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

Travis CI 是构建 R 包的常用工具。在我看来，在持续集成中使用 R 是最好的平台。一些下载量最大的 R 包就是在这个平台上构建的。例如*测试*、 *magick* 或 *covr* 。我也在这个平台上构建了我的包 [*RTest*](https://medium.com/@zappingseb/rtest-pretty-testing-of-r-packages-50f50b135650) 。在安装过程中，我遇到了一些麻烦。在这本指南中，我将与你分享我获得的知识。

# 目录

*   [来自“构建 R 项目”的基础知识](#a056)
*   [修改 R CMD 版本](#031a)
*   多种操作系统
*   [使用用户界面运行脚本](#45c4)
*   [代码覆盖率](#5da6)
*   [构建并部署一个 pkgdown 页面到 github 页面](#22c9)
*   [ImageMagick 和 Travis CI](#7ef5)
*   [延伸阅读](#b418)

# “构建 R 项目”的基础知识

Travis CI 的文章“[构建一个 R 项目](https://docs.travis-ci.com/user/languages/r/)”告诉您一些基本知识。它允许为 R 包或 R 项目建立一个构建。主要的收获来自这个. travis.yml 文件。

```
# Use R language
language: r#Define multiple R-versions, one from bioconductor
r:
  - oldrel
  - release
  - devel
  - bioc-devel# Set one of you dependencies from github
r_github_packages: r-lib/testthat# Set one of your dependencies from CRAN
r_packages: RTest# set a Linux system dependency
apt_packages:
  - libxml2-dev
```

教程向你解释你应该把你的类型语言设置成 **R.** 你可以使用不同的 **R 版本。**这些 R 版本是:

[旧版本，发布，开发，生物开发，生物发布]

此外，你可以通过`r_github_packages`从 github 加载任何包。或者你可以通过`r_packages`从[起重机](https://cran.r-project.org/)处拿到任何包裹。可以使用标准的 yml 格式创建多个包的列表:

```
r_packages:
  - RTest
  - testthat
```

如果你有 Linux 依赖，这需要被提及。RTest 包使用 XML 测试用例。需要的 XML Linux 库是`libxml2`。可以通过以下方式添加:

```
apt_packages:
  - libxml2-dev
```

您已经完成了基本操作。如果您的存储库中有这个. travis.yml 文件，它将使用`R CMD build`和`R CMD check`来检查您的项目。

# 修改 R CMD 命令

为了建立我的项目，我想把它建得像克兰一样。因此，我需要更改包检查的脚本。因此我补充道:

```
script:
  - R CMD build . --compact-vignettes=gs+qpdf
  - R CMD check *tar.gz --as-cran
```

在这个脚本中，你可以改变`R CMD build`或`R CMD check`参数。关于`R CMD`的参数列表，请参见 RStudio 的本[教程。](https://support.rstudio.com/hc/en-us/articles/200486518-Customizing-Package-Build-Options)

要运行晕影压缩，请通过以下方式获得`gs+qpdf`:

```
addons:
  apt:
    update: true
    packages:
      - libgs-dev
      - qpdf
      - ghostscript
```

# 多重操作系统

Travis CI 目前提供两种不同的操作系统(2019 年 1 月)。那些是 macOS 和 Linux。测试的标准方式是 Linux。对于我的项目 RTest，我也需要在 macOS 中进行测试。要在两个操作系统中进行测试，请使用 Travis CI 的`matrix`参数。

`matrix`参数允许为某些构建调整某些参数。为了在 Linux 和 macOS 中拥有完全相同的版本，我使用了以下结构:

```
matrix:
  include:
  - r: release
    script:
      - R CMD build . --compact-vignettes=gs+qpdf
      - R CMD check *tar.gz --as-cran 
  - r: release
    os: osx
    osx_image: xcode7.3
    before_install:
      - sudo tlmgr install framed titling
    script:
      - R CMD build . --compact-vignettes=gs+qpdf
      - R CMD check *tar.gz --as-cran
```

`matrix`函数将构建分成不同的操作系统。对于 macOS，我使用了 xcode7.3 图像，因为它是由 [rOpenSCI](https://ropensci.org/blog/2016/07/12/travis-osx/) 提出的。这个版本额外的一点是，它接近目前的 CRAN macOS 版本。正如你所看到的，你应该安装乳胶包`framed`和`titling`来创建插图。

# 使用用户界面运行脚本

我的包 [RTest](https://medium.com/@zappingseb/rtest-pretty-testing-of-r-packages-50f50b135650) 使用 [Tcl/Tk](https://www.rdocumentation.org/packages/tcltk/versions/3.5.2) 用户界面。为了测试这样的用户界面，您需要分别在 Linux 和 macOS 中启用用户界面。Travis CI 为 Linux 提供了`xvfb`包。对于 macOS，您需要用`homebrew`重新安装`xquartz`和`tcl-tk`。

## Linux 的用户界面

要在 Linux 中启用用户界面，请安装`xvfb`。

```
addons:
  apt:
    update: true
    packages:
       -   x11proto-xf86vidmode-dev
       -   xvfb
       -   libxxf86vm-dev
```

您可以在用户界面中运行所有的 R 脚本，方法是在`R`命令前使用`xvfb-run`命令。

```
script:
      - R CMD build . --compact-vignettes=gs+qpdf
      - xvfb-run R CMD check *tar.gz --as-cran
```

## macOS 的用户界面

对于 macOS 来说，用户界面的安装更加困难。您需要将`xquart`和`tcl-tk`添加到`xcode7.3`中提供的图像中。

```
before_install:
      - brew update
      - brew cask reinstall xquartz
      - brew install tcl-tk --with-tk
      - brew link --overwrite --force tcl-tk; brew unlink tcl-tk
```

要使用 xquartz，macOS 下没有`xvfb-run`命令。在 github 的一期[中，我发现了一个解决方案，仍然可以让用户界面与`xquartz`一起工作。](https://github.com/travis-ci/travis-ci/issues/7313)

```
before_script:
      - "export DISPLAY=:99.0"
      - if [ "${TRAVIS_OS_NAME}" = "osx" ]; then ( sudo Xvfb :99 -ac -screen 0 1024x768x8; echo ok ) & fi
```

在运行 R 会话可以使用的任何 R 脚本之前，创建一个显示。导出`DISPLAY`变量很重要。该变量由 tcktk R 包读取。

在 macOS 中，你不需要修改脚本

```
script:
      - R CMD build . --compact-vignettes=gs+qpdf
      - R CMD check *tar.gz --as-cran
```

## 插件

有关用户界面的更多信息，您可以阅读以下两个 github 问题:

*   [带特拉维斯和闪亮的硒元素](https://www.google.com/search?q=rselenium+shiny+trvais&rlz=1C1GCEA_enDE800DE801&oq=rselenium+shiny+trvais&aqs=chrome..69i57.2913j0j4&sourceid=chrome&ie=UTF-8)
*   [酱硒](https://github.com/ropensci/RSelenium/issues/145)

# 代码覆盖率

对于代码覆盖率，我建议使用一个特定版本的构建。我决定用 Linux + r-release 来测试代码覆盖率。首先，我将 *covr* 包添加到我的构建脚本中:

```
r_github_packages:
  - r-lib/covr
```

其次，我想使用 *covr* 测试我的包。这可以在 Travis 中使用`after_success`步骤完成。为了在这个步骤中使用 *covr* ，你需要定义你的包 tarball 将如何命名。您可以将这一点直接写入您的脚本。更好的方法是将其写入 you .travis.yml 文件的`env`部分。您的 tarball 的名称将始终是*package name+" _ "+package version+" . tar . gz "。在您的描述文件中，您定义了 *PackageName* 和 *PackageVersion。我使用 CODECOV 来存储我的覆盖测试的结果。**

```
env:
    - PKG_TARBALL=RTest_1.2.3.1000.tar.gzafter_success:
      - tar -C .. -xf $PKG_TARBALL
      - xvfb-run Rscript -e 'covr::codecov(type=c("tests", "vignettes", "examples"))'
```

我在我的包中使用的设置包括了我所有的例子、插图和测试的代码覆盖率。要部署代码覆盖率的结果，您必须定义全局变量`CODECOV_TOKEN`。令牌可以在`https://codecov.io/gh/<owner>/<repo>/settings`下找到。你可以把它秘密地插入你的崔维斯电脑里。在`https://travis-ci.org/<owner>/<repo>/settings`内添加代币。环境变量部分为您秘密存储变量。

要使用工作服而不是代码罩，使用`covr::coveralls`功能并在您的环境中定义一个`COVERALLS_TOKEN`。

# 构建并部署一个 pkgdown 页面到 github 页面

构建一个 *pkgdown* 页面对于记录您的代码非常有用。在我的 github 存储库中，我还托管了我的包 RTest 的 *pkgdown* 页面。你可以在这里找到页面:[https://zappingseb.github.io/RTest/index.html](https://zappingseb.github.io/RTest/index.html)

为了允许部署到 github 页面，我在 https://github.com/<owner>/<repo>/设置中激活了这个特性。你必须使用 gh-pages 分公司。如果您没有这样的[分支，您需要创建它](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/)。</repo></owner>

在. travis.yml 中，首先安装 *pkgdown。*

```
r_github_packages:
  - r-lib/pkgdown
```

你将不得不从你的包 tarball 建立页面。必须定义包 tarball 的名称。请参见章节[代码覆盖率](#5da6)了解如何做到这一点。拆开 tarball 的包装后，您应该在`rm -rf <PackageName>.Rcheck`之前删除检查包装时留下的任何内容。

```
after_success:
      - tar -C .. -xf $PKG_TARBALL
      - rm -rf RTest.Rcheck
      - Rscript -e 'pkgdown::build_site()'
```

`Rscript`将在`docs`文件夹中生成网站。该文件夹必须部署在 github 页面上。

登录 github 后，首先进入[https://github.com/settings/tokens](https://github.com/settings/tokens)。在那里，您必须创建一个具有`public_repo`或`repo`范围的令牌。现在将这个令牌存储在您的 Travis CI 构建中。因此转到`https://travis-ci.org/<owner>/<repo>/settings`并将其存储为名为`GITHUB_TOKEN`的全局变量。现在，将使用以下脚本在每个成功的构建上部署网站:

```
deploy:
  - provider: pages
        skip-cleanup: true
        github-token: $GITHUB_TOKEN
        keep-history: false
        local-dir: docs
        on:
          branch: master
```

有关部署页面的更多信息，您可以查看第页的 [Travis CI 指南。](https://docs.travis-ci.com/user/deployment/pages/)

# ImageMagick 和 Travis CI

在 [travis-ci 社区](https://travis-ci.community/t/error-configuration-failed-for-package-magick/1674)中，有一个关于如何在 travis-ci 上安装`magick`包的问题。答案很简单。您需要拥有 ImageMagick 的所有系统依赖项。通过以下方式为 Linux 安装这些软件:

```
addons:
   apt:
     update: true
     sources:
       - sourceline: 'ppa:opencpu/imagemagick'
       - sourceline: 'ppa:ubuntugis/ppa'
     packages:
       - libmagick++-dev
       - librsvg2-dev
       - libwebp-dev
       - libpoppler-cpp-dev
       - libtesseract-dev
       - libleptonica-dev
       - tesseract-ocr-eng
       - r-cran-rgdal
       - libfftw3-dev
       - cargo
```

这也适用于我的 macOS。

*亲爱的读者:写我在持续集成方面的工作总是令人愉快的。感谢你一直读到这篇文章的结尾。如果你喜欢这篇文章，你可以在* [***中***](https://medium.com/p/c977015bc6a9) *或者在*[***github***](https://github.com/zappingseb/RTest)*上为它鼓掌。如有评论，请在此处***或在我的****LinkedIn****个人资料*[*http://linkedin.com/in/zappingseb.*](http://linkedin.com/in/zappingseb.)上留言*

# *进一步阅读*

*   *[在 Travis CI(基础)上构建 R 项目](https://docs.travis-ci.com/user/languages/r/)*
*   *[Travis CI 社区为 R](https://travis-ci.community/c/languages/r) (论坛)*
*   *[RTest 包. travis.yml 文件](https://github.com/zappingseb/RTest/blob/master/.travis.yml)*
*   *[RTest 包—Travis-CI 的 99%代码覆盖率](/rtest-pretty-testing-of-r-packages-50f50b135650)*