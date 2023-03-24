# Linux 及更高版本的 Web 服务器

> 原文：<https://towardsdatascience.com/web-servers-in-linux-and-beyond-1241ca557a4f?source=collection_archive---------22----------------------->

Web 服务器似乎是一个难以理解和令人困惑的话题，尤其是当它与像 Linux 这样的外来操作系统结合在一起时。除了最初的复杂性之外，web 开发的世界总是在变化的，你一年(或一个月)学到的东西。)下一次可能不再有效。此外，没有一种特定的 web 服务器技术，初学者可能会发现很难置身于不同的阵营和框架中。然而在实践中，web 服务器的理论和应用是简单和用户友好的。在这篇文章中，我将介绍什么是 web 服务器，以及它们在 web 开发中的不同应用。

![](img/6ad500120894eb0988f26369551c6824.png)

Photo by [Kelvin Ang](https://unsplash.com/@kelvin1987?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

web 服务器是将文件从一台计算机传输到另一台计算机的工具。文件要去的地方被称为客户端，(通常是个人电脑上的浏览器)，而服务器本身也是一台电脑，尽管它没有屏幕或传统的 GUI。虽然我们大多数外行人认为计算机与我们用它们做什么有关，但计算机有它们自己看不见的重要功能，例如服务器。服务器的工作是托管网站，客户端(您)可以请求和访问这些网站。总之，web 服务器监听来自客户机的请求，并向它们返回数据，通常是一个文件。

Web 服务器托管网站，网站只是文件的集合，(尽管文件的扩展名类型很奇怪，如。html，。css，还有。js)。当您想要访问一个网站时，您键入它的 URL 或统一资源定位符，您的计算机就连接到托管它的服务器。这种对服务器上文件的请求称为 HTTP 请求，代表超文本传输协议。如果服务器找不到该文件，将返回 404 响应。需要记住的一件重要事情是，一个网页的不同组件可能位于不同的服务器上；例如，网页上的广告可能来自一个服务器，而图像和视频可能来自另一个⁴.

如上所述，不存在放之四海而皆准的 web 服务器系统，但是有各种围绕特定技术的框架，称为软件包。其中一个框架是围绕 Apache(世界上使用最广泛的 web 服务器软件)和另外两个经常与 Apache 一起使用的工具(PHP 和 MySQL)构建的。PHP 是一种服务器端脚本语言，对于设计网站非常重要，而 MySQL 是一种存储在线 data⁶.的流行数据库这三个工具与上面提到的 Linux、Windows 和 Mac 操作系统相结合，分别形成 LAMPstack、WAMPstack 和 MAMPstack。虽然它们可以在所有的操作系统上工作，但它们的安装简易性各不相同。Apache 预装在许多 Linux 发行版(或 distros)⁵和 Mac OS-X)上。Windows 也可以运行 Apache web 服务器，但是安装它的过程更加 complicated⁷.化虽然 Lamp 和 Wampstack 一直是 web 开发的主流软件包，但是最近出现了一些以 JavaScript 为中心的替代软件，我将在下面介绍。

JavaScript 长期以来被认为只是一种客户端语言，但近年来它已经成为自己的 web 服务器软件包的基础，被称为 MEANstack⁸.MEANstack 的组件是 MongoDB，一个非关系数据库程序，它使用类似 JSON 的 objects⁹，Express，一个后端 web 应用程序框架，Angular，一个前端 web 应用程序框架，以及 Node.js，一个用于开发基于 JavaScript 的应用程序的软件。MEANstack 的一个好处是它在整个应用程序中只使用一种语言，不像 Lampstack 必须在多种语言⁰.之间切换 MEANstack 为传统的基于 LAMPstack 的 web 服务器提供了一种替代方案，但是这两种技术都有可能在未来的 web 开发中扮演重要的角色。

网络服务器是现代互联网的主干，它让我们能够跨越遥远的距离获得新的想法和信息。虽然 web 服务器的术语和使用可能会令人困惑，但该领域技术的快速变化是令人放心的，因为这意味着其他人可能也和您一样困惑！随着这项技术的不断变化和进步，要保持与时俱进所需要的就是学习和掌握谷歌。只有这两种技能，你会发现自己在未来的几年里都在设计网站。

1.  [https://www . lifewire . com/servers-in-computer-networking-817380](https://www.lifewire.com/servers-in-computer-networking-817380)
2.  【https://www.sitepoint.com/how-to-install-apache-on-windows/ 号
3.  【https://likegeeks.com/linux-web-server/ 
4.  [https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview](https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview)
5.  https://www.wpbeginner.com/glossary/apache/
6.  [http://www . linuxandubuntu . com/home/how-to-setup-a-web-server-and-host-website-on-your-own-Linux-computer](http://www.linuxandubuntu.com/home/how-to-setup-a-web-server-and-host-website-on-your-own-linux-computer)
7.  [http://www.wampserver.com/en/](http://www.wampserver.com/en/)
8.  [http://mean.io/](http://mean.io/)
9.  [https://dzone.com/articles/comparing-mongodb-amp-mysql](https://dzone.com/articles/comparing-mongodb-amp-mysql)
10.  [https://www.codingdojo.com/what-is-the-mean-stack](https://www.codingdojo.com/what-is-the-mean-stack)