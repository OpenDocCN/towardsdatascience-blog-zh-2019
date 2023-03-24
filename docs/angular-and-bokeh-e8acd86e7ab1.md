# 角度和散景

> 原文：<https://towardsdatascience.com/angular-and-bokeh-e8acd86e7ab1?source=collection_archive---------12----------------------->

我们最近遇到了在应用程序或网站中显示质量图表的问题。但是除此之外，您还希望能够从 python 后端发送更新事件，并拥有所有漂亮的交互，比如按钮按压和文本输入事件，对吗？

![](img/472c4179d21299e3263712d7f6ca15d1.png)

散景图组件可能并不总是最佳的解决方案，但是尽管如此，我们希望在 GitHub 知识库中与您分享我们认为是一个很好的、最小的例子，演示了如何将 python 后端集成到 angular 应用程序中。

## 起点

是我们集成到 Angular 项目中的 BokehJS 库。绘图数据由 websocket 服务提供，在我们的示例中，我们使用 aiohttp，但是您可以设置任何其他 websocket 连接。角度组件可以通过它的标签名集成到 html 中的任何地方，下面的代码片段显示了散景图组件

```
<bokeh-chart></bokeh-chart>
```

散景图组件是一个规则的角度组件，有一个 html 部分

```
<div [id]="id"></div>
```

和打字稿部分。图表组件只需要向它自己的 html 部件提供 id。图表的数据由一个服务提供，这个服务在 **ngOnInit** 中的组件初始化时被调用。散景图组件的相关 typescript 部分如下所示

```
...
export class BokehChartComponent implements OnInit {
  public id: string;

  constructor(
    private bokehService: BokehService) { }

 ngOnInit() {
     this.id = "chart";
     this.bokehService.getChart(this.id);
 }
}
```

由于 BokehJS 库没有可用的类型，angular 中的集成并不像它应该的那样简单。人们只能通过库的全局暴露对象来访问库，在这种情况下，它也被命名为 **Bokeh** ，并且它是嵌入图表所必需的唯一挂钩。

```
// this is the global hook to the bokehjs lib (without types)
declare var Bokeh: any;
```

只有当你将普通的 java 脚本插入 angular 应用程序**index.html**最顶层的 html 文件时，这种魔力才会如预期的那样发挥作用

```
<head>
 ...
  <link
    href="[https://cdn.pydata.org/bokeh/release/bokeh-1.0.4.min.css](https://cdn.pydata.org/bokeh/release/bokeh-1.0.4.min.css)"
    rel="stylesheet" type="text/css">
  <script src="[https://cdn.pydata.org/bokeh/release/bokeh-1.0.4.min.js](https://cdn.pydata.org/bokeh/release/bokeh-1.0.4.min.js)"></script>
 </head>
```

## 博客服务

通过 **MessageService** 为图表提供数据，MessageService 封装了与后端的连接，因此只需通过公开的方法 sendMsg(msg)发送适当格式的消息。

```
export class BokehService extends Connector {constructor(private msgService: MessageService) {
 super(‘BokehService’);
 this.msgService.register(this);
 }…public getChart(id: string) {
 const msg = {
 name: ‘addChart’,
 args: [id],
 action: ‘default’
 };
 this.msgService.sendMsg(msg);
 }
```

该服务还向后端公开了一个方法，该方法实际上将图表绘制到原生 DOM 元素中，我们首先必须删除之前的绘图。

```
public plot(msg: Message) {
      const id = msg.args.id;
      const el = document.getElementById(id);
      // first remove the previous charts as child
      // like this, bokeh does not let us update a chart
      while (el.hasChildNodes()) {
            el.removeChild(el.lastChild);
      }
      // be sure to include the correct dom-id as second argument
      Bokeh.embed.embed_item(msg.args.item, id);
    }
```

## 后端服务

在我们的例子中是用 python 写的。我们使用 aiohttp 作为 web 服务器的异步解决方案。在浏览器中启动 angular 应用程序后，angular WebsocketService 立即连接到服务器端的 python 后端。请记住，在生产中，您将在这一点上实现更多的安全性，比如身份验证。后端准备接收来自 angular 的事件，例如给我散景图的数据。

来自 angular 的消息调用的 **addChart** 将 chartItem 作为连接到 websocket 服务的 json 项发送

```
 async def addChart(self, id_, user):
        """
        Example for adding a bokeh chart from backend

        """
        chartItem = self.chartProvider.chartExample()
        print("try to add chart for dom-id %s" % id_)
        context = {"name": "BokehService",
                   "args": {"item": chartItem, "id": id_},
                   "action": "plot"}
        await self.send_event(json.dumps(context), user=user)
```

这里有趣的部分是 send_event 方法，它实际上是基于我们的 websocket 服务器的实现。如前所述，在您的具体实现中，该部分可能会有所不同。

图表的最小示例也是作为 **ChartProvider** 类的成员函数编写的，它看起来非常简单，只为散景中的普通正弦图生成数据

```
import time
import numpy as np
from bokeh.plotting import figure
from bokeh.embed import json_itemclass ChartProvider(): def chartExample(self):
        t0 = time.time()
        # prepare some data
        self.phi += 0.02
        x = np.arange(0., 10., 0.1)
        y = np.sin(x + self.phi)
        # create a new plot
        p = figure()
        p.line(x, y, legend="SIN")
        chart_item = json_item(p)
        print(time.time()-t0)
        return chart_item
```