# 使用 Kibana 和 Elasticsearch 分析蜜罐数据

> 原文：<https://towardsdatascience.com/analysing-honeypot-data-using-kibana-and-elasticsearch-5e3d61eb2098?source=collection_archive---------12----------------------->

oneypots 是一个有趣的数据来源，应该用来加强商业网络战略。一些公司已经部署了镜像一些核心服务的蜜罐，只是为了监控黑客正在进行何种攻击，而其他公司正在使用蜜罐来丰富其他系统并建立系统规则*(即，如果 IP 出现在我们的蜜罐数据中，在我们的防火墙上拦截)*。然而，对于学生来说，使用蜜罐数据来了解真实的网络攻击可能是进入日志分析、机器学习和威胁情报的 f̵u̵n̵好方法。我最近发表了一篇关于[如何使用 Elastic Stack](https://medium.com/@Stephen_Chap/deploying-monitoring-honeypots-on-gcp-with-kibana-899fef6fdf76) [(Elasticsearch，Logstash & Kibana)](https://www.elastic.co/) 部署蜜罐的帖子，在两周的时间里，蜜罐已经在 GCP(谷歌云平台)——欧盟地区上线，在这篇帖子中，我们将了解它所遭受的所有攻击，威胁者是谁，以及我们如何利用这些数据。

## 蜜罐下面是什么？

*   Debian 10 ( [Buster](https://www.debian.org/News/2019/20190706) ) — 2 个 vCPUs，9GB RAM & 40GB 存储。
*   蜜罐—[T-Mobile 的 Tpot](https://github.com/dtag-dev-sec/tpotce)
*   Elastic Stack: Elasticsearch，Logstash & Kibana 版本 6.4 (D̳o̳c̳k̳e̳r̳s̳)

T-Pot 是用于此分析的一个很好的部署，因为它包括从电子邮件服务器到 RDP 服务器的 15 个不同的蜜罐系统，这确保了我可以获得足够大的数据池。

运行在 T-Pot 中的蜜罐的完整列表:

*   [](https://github.com/huuck/ADBHoney)
*   **[](https://github.com/Cymmetria/ciscoasa_honeypot)**
*   ***[](http://conpot.org/)***
*   ***[](https://github.com/cowrie/cowrie)***
*   ***[](https://github.com/DinoTools/dionaea)***
*   ***[](https://github.com/schmalle/ElasticpotPY)***
*   ***[*饕餮*](https://github.com/mushorg/glutton) *，****
*   **[*预示着*](https://github.com/johnnykv/heralding) *，***
*   **[](https://github.com/foospidy/HoneyPy)**
*   ***[*美人计*](https://github.com/armedpot/honeytrap/) *，****
*   **[](https://github.com/awhitehatter/mailoney)**
*   ***[](https://github.com/schmalle/medpot)***
*   ***[*【rdpy】*](https://github.com/citronneur/rdpy)***
*   ***[*圈套*](http://mushmush.org/) *，****
*   **[*坦纳*](http://mushmush.org/)**

**由于数据是在 Elasticsearch 中收集的，仪表盘是可用的，但我也可以构建自己的仪表盘来更好地理解数据。如果你以前从未设置过 Elasticsearch 集群，我推荐你阅读并尝试一下 [Digital Ocean 关于设置 Elastic Stack](https://www.digitalocean.com/community/tutorials/how-to-install-elasticsearch-logstash-and-kibana-elastic-stack-on-ubuntu-16-04) 的教程。**

## **弹性集群分析📉**

**由于这是一个运行 6.8.2 版本的单节点设置，我们无法访问机器学习*(需要黄金/白金许可证，但有 30 天的试用期，您可以试用一些功能，或者可以试用 2 周的弹性云)。*但是，我们确实使用了第三方工具 [Elasticsearch Head](https://mobz.github.io/elasticsearch-head/) ，它为我们提供了集群运行情况的概述。**

**![](img/9e5168b80a13c6fb938adaeb86d13572.png)**

**The Elastic Cluster**

## **那么，这一切的意义何在？**

**威胁情报是简单的答案。威胁情报(Threat intelligence)或网络威胁英特尔(cyber threat intel)简称(看起来更长)被描述为，*“一个组织用来了解* ***已经、将要或当前*** *针对该组织的威胁的信息。”*使用这个标准定义和我们的蜜罐数据，我们可以看到我们有一个关于正在进行的攻击、谁在进行攻击以及他们针对什么端口的数据池。这些数据可用于生成英特尔报告，这些报告可输入到 **SOAR** 系统或用于生成报告。知道谁是持久的攻击者也会帮助你抵御他们的攻击，你对敌人了解得越多越好，对吗？对于 EWS(预警系统)来说，什么是更好的数据源？**

## **翱翔？**

**安全协调、自动化和响应正在成为事件响应和管理 SIEM 平台的重要组成部分。比如 [Splunk](https://www.splunk.com/en_us/form/the-soar-buyers-guide.html?utm_campaign=google_emea_tier1_en_search_brand&utm_source=google&utm_medium=cpc&utm_content=Soar_buyersguide_guide&utm_term=%2Bsplunk%20%2Bphantom&_bk=%2Bsplunk%20%2Bphantom&_bt=310388742139&_bm=b&_bn=g&_bg=68666554984&device=c&gclid=EAIaIQobChMI0KX6gpns5QIVC7DtCh10XgM8EAAYASAAEgJ8GPD_BwE) 的用户会对 [Phantom 比较熟悉。](https://www.splunk.com/en_us/software/phantom.html)一个很好的有社区版的 SOAR 平台是[德米斯图拉。](https://www.demisto.com/)可以将数据从您的蜜罐弹性集群直接馈送到连接到 SIEM 的 SOAR 平台。您可以将其用作事件的丰富工具，或者搜寻零日攻击。蜜罐给你的是持续不断的“真实”攻击和恶意行为者，随着你走向自动化决策，访问真实的威胁数据将有助于丰富你的威胁情报。**

## **最后但同样重要的是，数据！📊**

**正在考虑构建一个数据湖？数据越多越好，对吗？即使你是一个渗透测试人员，想要更新你的密码列表，拥有一个收集最新用户名和密码组合的蜜罐意味着如果有任何数据泄漏和凭据被使用，你的蜜罐将让你访问数据。**

**![](img/7d4ba76e51976ff57deec80a1232320b.png)**

**Shout out to everyone still using admin/admin credentials!**

## **让我们打开蜜罐吧！🍯**

**由于部署的系统已经运行了一段时间，我想看看已经发生的攻击，并调查一些恶意威胁行为者。由于数据在 Elasticsearch 中，可以通过 Kibana 查看，我们不需要抓取任何日志文件，这本身就是一个胜利。**

## **最常见的攻击是什么？**

**![](img/c4a0092e621fb566dc6b617e1f0f9266.png)**

**The Cowrie honeypot had over 100k attacks!**

**Telnet 和 SSH 蜜罐受到的攻击最多也就不足为奇了。Cowrie 是一个中高交互蜜罐，旨在吸引和记录攻击者的暴力攻击和任何外壳交互。它的主要目的是与攻击者互动，同时监控他们在认为自己已经入侵系统时的行为。你可以在这里阅读更多关于 T2 的内容。**

**毫不奇怪，在 102，787 起*攻击中， ***32，607 起*** 攻击来自中国，美国以适度的 ***14，115*** 攻击位居第二。***

**![](img/093895647ad84668caf505811e7e361a.png)**

**Top 10 Attacking nations**

**Cowrie 标记任何已知的攻击者、大规模扫描仪、不良信誉攻击者、垃圾邮件机器人和 tor 出口节点，在 ***102，787****99.95%*的攻击中，这些都是已知的攻击者，之前已经报告过恶意活动。我决定调查攻击蜜罐最多的 IP(*你好，128.199.235.18🙃*)并使用 [AbuseIPDB](https://www.abuseipdb.com/check/128.199.235.18) 我们可以看到，该 IP 因各种恶意活动已被报告超过 928 次。**

**![](img/2ded112498c90f064177e9f0ab79f067.png)**

**This server is hosted on Digital Ocean (I notified them)**

## **但是他们登录后会做什么呢？**

**![](img/9c3e126c4866bdca4598d97715100dfa.png)**

**Top 10 attacker input**

**了解攻击者在获得访问权限后正在做什么是构建防御的关键。并非所有攻击都来自来自未知 IP 的未知用户。对于大多数 SIEMs，如果用户使用任何列出的命令，都有可能触发警报。这样，如果您网络上的用户丢失了他们的凭证，如果使用了其中的任何命令，您可以将其标记为可疑行为，并设置规则来阻止帐户，直到确认这不是违规行为。**

> **注意:不要阻塞' top '命令，我们仍然需要它！😭**

# **并不总是通常的嫌疑人！**

**![](img/24e2f9c0fe4b3140254a77c4a6afb493.png)**

**A map where all the attacks have come from**

**网络攻击不仅仅是四大(俄罗斯、中国、美国和伊朗)。对于安全团队来说，理解这一点是制定策略的关键。定期检查您的服务在哪里被访问/攻击也有助于确定是否有任何内部破坏正在进行。**

**这篇文章的主要目的是向网络安全爱好者介绍蜜罐背后发生的事情，数据以及你可以用它做什么。如果您对部署自己的蜜罐感兴趣，可以看看我的教程，了解如何设置我使用的部署。然而，我确实注意到，一些蜜罐码头工人在一次小的暴力攻击后一直在敲门，我单独设置了一个弹性集群来监控服务器，看它是否受到攻击，事实确实如此。我的下一篇文章将介绍如何使用最新版本的 Elastic 来使用 SIEM 功能监控一些测试环境。我还计划清理蜜罐，并单独部署它们以进行更好的分析，并利用来自 Elastic 的机器学习。**

**![](img/29a488f90bc9c376f5aae74f7f08cfe0.png)**

**RDPY Attacks — Not all data has to be on Excel!**

**斯蒂芬·查彭达玛**