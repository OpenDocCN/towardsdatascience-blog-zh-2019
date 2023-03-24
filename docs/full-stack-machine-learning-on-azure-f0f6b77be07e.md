# Azure 上的全栈机器学习

> 原文：<https://towardsdatascience.com/full-stack-machine-learning-on-azure-f0f6b77be07e?source=collection_archive---------14----------------------->

**使用 MERN 在 Azure 上构建全栈机器学习网站指南(Mongo，Express，React (Node.js))**

![](img/786d0ea4b7273eed2f5513c4b3fd6716.png)

Tech stack and flow for [http://diamonds.foostack.ai](http://diamonds.foostack.ai)

在对最新技术感到生疏后，我决定同时学习一系列东西，并开发了[*http://diamonds . foo stack . ai*](http://diamonds.foostack.ai/)，现在已经上线。(*基础机器学习 app 给钻石定价*)。

我的女朋友让我给她买一枚钻戒，我告诉她“不，等等！！！我必须先建立一个定价应用程序/网站……(*在家试试*)”

本博客中的步骤和章节概述:

1.  使用 Python 获取数据
2.  使用 ScikitLearn 在 Jupyter 笔记本电脑中培训 ML 模型
3.  将 ML 模型部署到 Azure
4.  在 React w/ Express 上构建前端
5.  构建后端 Express，MongoDB
6.  Azure Web 应用部署和 CI/CD
7.  总结、提示和技巧

回顾我的 github 签到记录，这个项目花了我 1 个月(26 天)——可能每周花 8-10 个小时，所以大概 40 个小时以上。最终产品还不算太差:

![](img/a6115b59531cdb8c99e483972ecb608a.png)

[http://diamonds.foostack.ai](http://diamonds.foostack.ai)

# 步骤 1:获取数据

我在网上搜索好的钻石数据。我发现:

*   [Kaggle 的钻石数据集](https://www.kaggle.com/shivam2503/diamonds)有 50k 行，但与我找到的其他数据集进行交叉比较时，它似乎缺乏真实世界的定价准确性。
*   其他“[实验数据集](https://datasciencereview.com/case-study-does-the-size-of-diamonds-affect-its-price -- https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Diamond.csv)”是旧的或者类似地与当前价格不同步。
*   一些商业供应商出售更新的钻石库存/价格数据，但我不愿意付钱。
*   零售网站有数据，但需要搜集(bluenile.com，rarecarat.com 等)

**解决:**被爆料加强[刮不违法](https://thenextweb.com/security/2019/09/10/us-court-says-scraping-a-site-without-permission-isnt-illegal/) —我在刮**BLUENILE.COM**——(感谢[安德鲁·马尔德](https://amarder.github.io/diamonds/)的首发码)。

![](img/2e03aa04d761fd2bbdfb2d4ac3d06cad.png)

My source of data!

安德鲁斯的代码基本有效，除了:

*   它是用 Python 2.7 写的，所以有一些问题
*   自从他写了以后，网站上的数据格式改变了
*   Bluenile 不断发展，目前已经阻止了运行此类脚本的吸血鬼

我唯一的黑客/贡献是在他们一直关闭我的时候添加重试循环:

```
# (bluenile API restricts to 1,000 page chunks at a time)
home = requests.get('http://www.bluenile.com/')
while True:
   url = 'http://www.bluenile.com/api/public/diamond-search-grid/v2'
   try:
      response = requests.get(url, params, cookies=home.cookies)
   except:  # request api exception, usually timeout
      time.sleep(60*30)
      next
   if (response.ok == False):  # server may have disconnected us
      time.sleep(60*30)
      next
   else:   # all going well
      time.sleep(60*4)
      next
```

最初的脚本进行了内联清理——我选择了下游清理(通常是个坏主意)。

输出示例 1 行(注意需要清理的＄逗号引号等):

```
carat,clarity,color,culet,cut,date,dateSet,depth,detailsPageUrl,fluorescence,hasVisualization,id,imageUrl,lxwRatio,measurements,polish,price,pricePerCarat,sellingIndex,shapeCode,shapeName,skus,symmetry,table,v360BaseUrl,visualizationImageUrl,willArriveForHoliday
['0.23'],['FL'],['D'],['None'],"[{'label': 'Ideal', 'labelSmall': 'Ideal'}]",['Sep 30'],['Oct 1'],['58.8'],./diamond-details/LD12750672,['None'],[True],LD12750672,,['1.01'],"[{'label': '4.04 x 4.00 x 2.36 mm', 'labelSmall': '4.04 x 4.00 x 2.36'}]",['Excellent'],871,"['$3,787']",0.7348354,['RD'],['Round'],['LD12750672'],['Excellent'],['60.0'],https://bluenile.v360.in/50/imaged/gia-7313767639/1/,https://bnsec.bluenile.com/bnsecure/diamondvis/gia-7313767639/still_230x230.jpg,False
```

最后，我的抢劫成功了 14 万颗圆钻！我将这些保存到一个 CSV 文件中，用于清理和模型训练。

# **第二步:用 Jupyter 和 SKLearn 训练模型**

Jupyter 笔记本很棒(看我的[全笔记本](https://nbviewer.jupyter.org/github/dougfoo/machineLearning/blob/master/diamonds/Diamond-Analysis-4.ipynb)这里):

*   当我试验不同的想法时，用简单的方法来注释我的想法——有助于以后修改或传递给其他人！
*   比 VS 代码项目设置和运行更快——隔离细胞和独立运行的能力就像修改过的 REPL！
*   可视化更容易操作，因为您可以重新运行单元，只绘制图形，而不是重新运行整个程序！
*   *(上面的一些可以用调试器和 REPL shell 来完成，但是看起来更干净&更容易在笔记本上重现)*

对于数据清理，一些选项:

1.  通过 python 脚本清理:CSV -> scrub.py -> CSV
2.  在笔记本内拖动:CSV ->笔记本-> pandas/python -> CSV
3.  在数据库内部清理:CSV-> MongoDB-> SQL scrubbing-> CSV
4.  使用其他工具，如 Alteryx、Trifacta 或 Informatica

选项#1 可以工作，但是为什么还要处理另一个脚本/env。#2 看起来是正确的方法，因为我们在 ML 之后使用 Python 和笔记本。#3 听起来我们在推迟不可避免的事情，而不是把垃圾放入数据库。#4 对像我这样的穷人来说不是一个真正的选择。

**使用熊猫进行擦洗&数据帧**

熊猫是令人惊奇的东西。加载我的 CSV 很容易，数据帧上的 describe()显示了一些基本的统计数据:

```
**import** **pandas** **as** **pd**
diamonds5 = pd.read_csv('./blue-nile-download-round-full.csv')
diamonds5.describe()
```

![](img/e7d7f6ad0f11d82b977922eadd1ab835.png)

describe() and head() of the pandas.Dataframe

如前所述，数据存在一些问题，这是我的数据相关培训前任务清单:

1.  确定在生产模型中使用哪些列
2.  将文本转换为干净的文本(即删除[])
3.  正确转换数字(删除$和，并删除[]和'')
4.  处理关键字字段中的空值或超出范围的值
5.  剔除异常值(使用≤ 4.5 克拉，因为此后会变得稀疏)

完成后，我们对克拉和切工(颜色)进行一点视觉图形检查，以确定价格——使用 Jupyter 非常简单:

```
fig, ax = plt.subplots()
**for** c, df **in** diamonds5a.groupby('cut'):
    ax.scatter(df['carat'], df['price'], label=c, s=np.pi*3)
fig.set_size_inches(18.5, 10.5)
```

![](img/e325b6d591c8e0389bd4d793a1211673.png)

Price to Carat (to Cut)

虽然形状和深度等一些指标可能对预测有价值，但我真的想只关注 4-C——克拉、净度、颜色和切割(加上 sku，这是 BlueNile 中的 ID)。我可以放弃剩下的。

*(注意，我发现烦人的 DataFrame 有一些 mutator 函数就地改变 obj，还有一些返回改变后的对象。似乎有点不一致？)*

```
# basic cleanup functions
def cleanBracketsToF(x):
    return float(cleanBracketsToS(x))def cleanBracketsToS(x):
    return x.replace("['",'').replace("']",'')def cleanCut(str):
    return str[str.index('label') + 9: str.index('labelSmall')-4]df.loc[:,'carat'] =   
    df.loc[:,'carat'].to_frame().applymap(cleanBracketsToF)
...*# clear nulls*
pd.set_option('use_inf_as_na', **True**)
df = df.loc[~df['carat'].isnull()]
...
```

在清理并仅包括培训所需的列后，我们有了这个(第一列是熊猫 DF 标识符，它是内部的，以及我们稍后为培训删除但稍后为显示/链接回 bluenile.com 而包括的 SKU):

![](img/bf24d1baee5c2c7b036d955ce2519122.png)

Pre column encoded

接下来，我们需要转换非数字(切割，颜色，清晰度)。您有几个编码选项:

1.  Onehot 编码(可能是最常见的，将类别转换为二进制列)
2.  顺序映射，如果它是分级的/连续的(例如，对于颜色，D->0，E->1，F->2 …)或者反过来(K->0，J-> 1…)

我用 sklearn 的 Pandas.get_dummies()调用去了 Onehot

```
cut = pd.get_dummies( diamonds5b['cut'], prefix='cut_')# result: cut__Astor Ideal', 'cut__Good', 'cut__Ideal','cut__Very Good'
# do the same for color, clarity...
```

**OneHot 编码**我们的数据现在看起来像这样:

![](img/741073b979ad2d2ef421a2ddbb1e193b.png)

Post OneHot encoded “cut”

我学到的一件棘手的事情是，如果你在一个 Hot 上对所有列进行编码，你可能会产生一个线性代数问题，这个问题是“奇异的”,这个小故事破坏了这个模型。阅读 [**这篇关于**的假人陷阱](https://www.algosome.com/articles/dummy-variable-trap-regression.html)。

这是将 one-hot 应用于所有列并测量系数的最终结果(将其视为每个特征的权重):

![](img/c4c52603fb16d3be76c8ae307a24a6fa.png)

Column coefficients of for a linear model

以下是 reg 中的准确重量。_coeff

```
coefficients: [ 1122.35088004  -957.70046613   471.56252273  3604.24362627
  3443.65378177  3128.96414383  2340.71261083  1717.39519952
  1673.9767721    590.1060328  10566.54997464  1383.32878002
 -1131.5953868  -1340.53213295  -457.74564551  -848.77484115
   540.46973926 15014.89306894]
intercept: -8389.974398175218
```

这意味着线性预测模型主要基于克拉(有道理)和完美无瑕的钻石。该等式的结果类似于:

```
y = -8389 + 1122*(cut == Astor ? 1: 0) + ….. + carat*15014)….
```

例如，如果我们为 1.0 克拉、切割理想、颜色 G、净度 VVS1 的钻石定价，计算结果为:

```
y = -8389 + 1*15014 (carats)+ 1*471 (cut)+ 1*2340 (color)+ 1*540 (clarity) = $9,976 
```

在 SKLearn 中训练一个模型——大部分代码是这样的。*警告——你可能会花很多时间调试错误的形状&数组&矩阵的方向！*

```
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)               # training
y_pred = regr.predict(X_test)            # inference
r2score = regr.score(X_test, y_test)     # get r2 score
```

R2 分数值得一提——它不是一个“准确性”分数，所以不要期望得到 99%。这是对准确性/差异的一种衡量，更多信息请点击@ [R2 维基](https://en.wikipedia.org/wiki/Coefficient_of_determination)。

我测试了其他的 [SKLearn 模型](https://scikit-learn.org/stable/modules/linear_model.html):岭回归、套索回归、保序回归(克拉上的单一特征)、弹性正则回归。较新的 AutoML 框架(H2O、DataRobot、Azure AutoML、SageMaker 等)同时测试所有模型&超参数..让很多业余 DS 失业..

接下来，我再次使用相同的数据集应用树和集成方法。尝试各种超参数组合后的最佳性能(R2)不足为奇 XGBoost:

```
est = GradientBoostingRegressor(n_estimators=ne, learning_rate=lr,
    max_depth=md, random_state=8, loss='ls').fit(X_train, y_train)
score = est.score(X_test, y_test)
```

![](img/b8e837abc7008b28d9805aefb6647f9d.png)

Rough feature weights w/ XGB

查看特征权重——克拉是最高的，其他具有中等影响(有趣的是，完美无瑕的清晰度不如线性回归有影响力)。

系数对于确定每个特征的“重要性”可能不是最佳的。 [Shapley 值](https://christophm.github.io/interpretable-ml-book/shapley.html)似乎更符合行业标准。

一旦在笔记本中进行了训练，我们就使用 SKLearn 的 joblib 库导出所有经过训练的模型，以便稍后进行部署(该库将文件“pickless”或序列化为专有格式([阅读 pickle 的一些不好的方面](https://www.benfrederickson.com/dont-pickle-your-data/)))。

```
joblib.dump(value=regr_full, filename='regr_model.pkl')
joblib.dump(value=model, filename='xgb_model.pkl')
joblib.dump(value=rfr, filename='randomforest_model.pkl')
joblib.dump(value=iso_reg, filename='isoridgelineartreg_model.pkl')
```

# 步骤 3:将模型部署到 Azure

退一步说，我们有几种方法可以在生产中运行模型:

1.  将预先训练好的模型嵌入后端。在我们的例子中，我们使用的是 Express (Javascript ),这不是最好的嵌入方式..
2.  将预先训练好的模型发布到某个地方(比如 Python Flask 容器),并通过 webservice 公开。
3.  使用像 Azure ML 这样的 API 发布和部署到他们的容器中。
4.  使用 Azure ML Studio 这样的框架进行设计、训练和发布(半自动)。

我们将做#2、#3 和#4

![](img/7d0d1b78b03d592186ccf0f1488eb68e.png)

第一步是在 Azure 上进行设置——如果这对你来说是新的，有几个基本步骤:

1.  在 portal.azure.com 上创建帐户
2.  你可以获得一段时间的免费订阅，但最终你可能需要创建一个付费订阅(小心$$$)。

# 创建新的 ML 服务工作场所

![](img/18dd8bf4512f512b1e56a9f74bce0660.png)

New stuff in Azure — evolving fast

我们可以使用新的 Azure ML API(尽管你应该看看 ML GUI 工作区——你可以启动 Jupyter 笔记本，尝试他们的菜单驱动 AutoML，ML Studio Visual GUI(下图),以及其他现在不断变化的东西……

![](img/ca21dfa880f91104206baa672d5e3f6e.png)

Drag-n-Drop ML… iffy distraction at best

# **通过 Python API 设置模型**

使用 API 在 Azure 上设置服务有几个步骤

1.  注册你的模型(你的泡菜。pkl]文件)
2.  设置环境文件和评分回调文件(。py 接口)
3.  设置 web 服务(AciWebService)、模型、推理配置和部署。

**注册**真的很简单，每个模型做一次(pkl 文件):

```
ws = Workspace.get(name='DiamondMLWS',
    subscription_id='****',resource_group='diamond-ml')new_model = Model.register(model_path="mymodel.pkl"model_name=name,
    description="mymodel descr",workspace=ws)
```

**Env** 文件可以用这样的脚本生成(一次):

```
myenv = CondaDependencies.create(conda_packages=['numpy',
   'scikit-learn'])
with open("myenv.yml","w") as f:
   f.write(myenv.serialize_to_string())
```

**评分**文件如下所示，为每个模型实现 init()和 run():

```
#score_mymodel.py
def init():
   global model
   model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'),
        'mymodel.pkl')
   model = joblib.load(model_path)def run(raw_data):
   data = np.array(json.loads(raw_data)['data'])
   y_hat = model.predict(data)
   return y_hat.tolist()
```

最后一步包括设置服务、模型，并为每个模型部署一次:

```
aciconfig = AciWebservice.deploy_configuration(cpu_cores=1,
   memory_gb=1,tags={"data": "diamonds","method": "sklearn"},
   description='Predict Diamonds with sklearn')
ws = Workspace.get(name='DiamondMLWS', subscription_id='****',
   resource_group='diamond-ml')
model = Model(ws, name)inference_config = InferenceConfig(runtime= "python",
   entry_script="score_mymodel.py", conda_file="myenv.yml")service = Model.deploy(workspace=ws,
   name=svcname, models=[model], inference_config=inference_config,
   deployment_config=aciconfig)
service.wait_for_deployment(show_output=True)
```

就是这样！您应该会看到输出声明服务已经部署并返回 URI 以供访问，并检查 Azure 门户以查看它是否在那里。

```
Running.......................
ACI service creation operation finished, operation "Succeeded"
http://dds2f-97f5eb5c4163.eastus.azurecontainer.io/score
```

差不多就是这样。输入格式和输出由您在第 2 部分中创建的模型细节定义。你可以用 Postman 或 curl 测试 API 确保它是定价的:

![](img/c979d6c39377f9f6147fadee4108845f.png)

Testing via POSTman

发布到 azure web service——价格 1.1 克拉，F 色，未切割，VVS2 净度钻石→ $11，272 是预测！

我不会深入讨论使用 GUI 驱动的 Azure ML Studio 的细节。您可以拖放、训练并最终发布一个 web 服务 URL(使用不同的输入/输出格式)。对于我的应用程序，我在 GUI ML Studio 中发布了 3 个模型，在 SKLearn 中发布了 4 个模型。查看 web GUI 上定价请求的输出:

![](img/aa715ad1a1a3c527ea72a79e0832226c.png)

Target aggregated results

请注意神经网络模型是完全疲惫不堪…不知道我是否会麻烦修复它…

由于 Azure ML 容器的成本，我选择将模型转移到 Azure 托管的 **Flask Python 服务**。这是非常少的代码——而且成本只是 Azure ML ACI 的一小部分($ 2/天每项服务，所以部署了 w/ 4 个模型，大概是$ 8/天..迅速破产的老道格·福】。重新部署非常容易。烧瓶中的 pkl:

```
app = Flask(__name__)
models = {}
models['XGB2'] = joblib.load('models/sklearn_diamond_xgb_model.pkl')
models['ISO'] = joblib.load('models/sklearn_diamond_iso_model.pkl')
models['LR3'] = joblib.load('models/sklearn_diamond_regr_model.pkl')print('loaded models', models)@app.route('/models/<model>', methods=['POST'])
def predict(model):
   if (models.get(model) is None):
      print('model not found: ',model)
      return jsonify("[-1]")
   j_data = np.array(request.get_json()['data'])
   y_hat = np.array2string(models[model].predict(j_data))
   print('input: ',j_data, ', results:', y_hat)
   return y_hatif __name__ == '__main__':
   app.run(debug=True)
```

# **步骤 4:用 React 构建完整的堆栈 ML 前端**

同样，关于启动 React 项目& Express.js 项目的故事数不胜数，所以我将跳过一些步骤..

*   将 **express** 安装到我的项目文件夹~/diamonds 中
*   安装 **react** ，用 create-react-app 在一个名为~/diamonds/diamond-app 的子目录中创建起始项目
*   对于生产部署，它们通过 webpack 合并成一个单一的 **express** 捆绑包

# React 与我以前使用过的所有其他框架相比

我对 Web 开发非常熟悉，比如 Perl-CGI、Php、JSP、ASP 和所有那些“Web 2.0”框架，包括传统的 AJAX。我也用过 Django，仔细观察过 Angular。 **React 不同——有一个学习曲线**。经过一番挣扎，我现在是你的忠实粉丝了！需要记住的关键事项:

*   **组件—** 编码使用 JSX 语言，看起来像 Javascript，但略有不同。你通过 JSX 类或函数来编码组件(<标签/ >)。顶层组件通常是 App.js，在这里你可以添加你的自定义 JSX 和其他组件，比如< MyTag/ >。

```
class MyTag extends React.Component {
    const constvar = "Text"
    render() { return (<div>Hi {constvar}</div>) }
}
```

*   **状态** —动态渲染流程与您可能习惯的非常不同。React 有自己的生命周期，主要由“状态”(特殊变量)变化控制。声明一组状态变量，然后像变魔术一样刷新受影响的组件。见下文，随着**动态**的值的改变，GUI 组件将自我刷新。

```
class MyTag extends React.Component {
    constructor(props) {
       super(props);
       this.state = { dynamic: 'Dyn txt' }
    }
    const constvar = "Const Txt";
    render() { 
       return (<div>Hi {constvar} and {this.state.dynamic}  </div>
    )}
}
// equiv as a function, bit tighter ?  also use MyTag
function MyTag2() {  
    const constvar = "Const Txt";
    const [dynamic, setDynamic] = React.useState('Dyn Txt');
    return (<div>Hi {constvar} and {dynamic} and <MyTag/> </div>);
}
```

*   **Props —** props 是标签属性和组件/标签的子组件—传递数据的一种方式。您可以通过 props 将数据 ***向下*** 传递，但是调用 ***向上*** 需要一个回调函数(例如在 caller/parent 中定义的 myCallbackFunc)。

```
function MyTag2(props) {  
  const constvar = "Const Txt";
  const [dynamic, setDynamic] = React.useState('Dyn Txt');
  return (
     <div> 
       Hi {constvar} and {dynamic} 
       Hi {props.var1} 
       <Button onClick={props.callback} />
     </div>);
  }
}
function App() {
  function myCallbackFunc() {
     // do something on button click
  }  return (
     <div> 
        <MyTag2 **var1**=’var' **callback**={myCallbackFunc}>Text</MyTag2> 
     </div>);
  }
}
```

上例——你的 app 根是<app>，在里面我们调用 MyTag **w/ "props"** var1 和回调以及内部的" Text "。</app>

这需要一些拨弄来学习，但它实际上非常优雅——试试吧！

**使用材料设计启动用户界面**

Google 的 Material-UI 是 React 的一个组件库/框架，它给你提供了很棒的小部件和默认主题，这样任何人都可以让网站看起来很棒！我浏览了画廊，选择了[仪表盘(在左边)](https://github.com/mui-org/material-ui/tree/master/docs/src/pages/getting-started/templates/dashboard)开始。

![](img/3fd86e75aeb562cbf15c4035ef9afff0.png)

Material-UI 有类似于<grid>布局和<paper>容器的组件来管理反应式/自动调整组件。包括一个漂亮的侧菜单<drawer>。</drawer></paper></grid>

我在默认仪表板上做的主要事情:

*   设置左抽屉默认关闭，添加<modal>帮助弹出链接到外部网站。</modal>
*   更新了顶栏和新文本，替换了通知铃声和帮助
*   在网格中实现了 3 个主要的“纸”容器:随机钻石、库存选择器/过滤器和 ML Pricer。

![](img/a8752c4185f6ced8463f8f0b7c1d029b.png)

# 今日随机钻石

![](img/4fc2d2b07f8d37b6a8730639e290bc0c.png)

这是最容易建立的一个。假设/diamonds/daily/上有一个服务器 REST 端点，它返回一个 json (data.price，data.carat，等等..)—我们只获取组件负载(useEffect())并从结果中设置状态变量。

```
export default function Daily() {
  const [diamond, setDiamond] = useState({
    price:0.0,carat:0.0,color:'',cut:'',clarity: '',skus: ''
  });  useEffect(() => {
    axios.get('/diamonds/daily/').then(response => {
      setDiamond({
         price: response.data.price, 
         carat: response.data.carat, 
         ...
      });
    })},
  []);  
  return (
    <React.Fragment><Title>Random Diamond Featured</Title>   
      <Typography>Price: {diamond.price}</Typography>
      <Typography>Carats: {diamond.carat}</Typography>
      ...
    </React.Fragment>
  );
}
```

# 钻石过滤器

![](img/d16c3fa9679e01e07fde2ff09da004c1.png)

我想要一个基本的搜索表单，让我通过 140k 钻石过滤-图表+表格中的细节。

我选择 [Victory](https://formidable.com/open-source/victory/docs/) 作为我的制图/绘图工具，因为它适用于反应式(也考虑了反应式和重新制图)。我不喜欢 material-ui 数据表(缺乏更高层次的抽象)，所以选择了更小的开源 [Material-Table](https://github.com/mbrn/material-table) 项目。在 React-land 中，您可以选择许多组件！

创建该组件的主要步骤:

1.  使用 Material-UI 表单组件，在表单组中添加复选框，并实现克拉滑块。
2.  实现 SearchButton 的 onClick
3.  模仿或构建您的[后端](https://medium.com/@doug.foo/full-stack-mern-machine-learning-on-azure-part-5-building-backend-express-mongodb-f87f5ddf707d)来返回一个列表中的钻石 JSON，我们在一个图形+表格中呈现(通过选项卡)— [更多内容在第 5 部分](https://medium.com/@doug.foo/full-stack-mern-machine-learning-on-azure-part-5-building-backend-express-mongodb-f87f5ddf707d)

参考 [github 获取完整源代码](https://github.com/dougfoo/diamonds/blob/master/diamond-app/src/Chooser.js)。下面显示了如何将选项传递给服务器:

```
class Chooser extends Component {
  constructor(props) {
  super(props);  this.state = {
    D: true,
    E: true,
    F: true,
    G: true,
    H: true,
    AstorIdeal: false,
    Ideal: true,
    diamonds: [],
    ...
  }  handleSubmit = event => {
    // call REST api, setState of return diamonds[] etc
  }  render() {
    return (
      <React.Fragment >
         <Title>Filter Diamonds</Title>
         <FormControl component="fieldset" >
           ... checkboxes and button 
         </FormControl>
         <Paper square className={classes.root}>
           <Chart diamonds={this.state.diamonds}/>
           <DiamondsTable diamonds={this.state.diamonds} />
         </Paper>
      </React.Fragment>
   )
}
```

DiamondsTable 是一个第三方表格组件[来查看数据网格。把钻石[]当道具传就行了。钻石[]状态的任何变化都会强制重新渲染。反应刷新魔法！](https://github.com/mbrn/material-table)

![](img/f6d3509e67564ddef86d20cd7f6691a5.png)

Tab to DiamondTable (customized react-table component)

# **ML 定价—终于！**

我们已经接近目标—根据我们训练集中的 14 万颗钻石预测钻石价格。

![](img/f4b53cdd9833b13c61838513fa1ddc71.png)

控件非常简单——很像过滤器/选择器，我们有一个基本的表单和按钮。

代码结构非常相似:

1.  带有 submitPrice 回调的 FormGroup 和 Button
2.  submitPrice 调用 REST 服务，传递输入{克拉、净度、切工、颜色}并返回设置为状态变量的钻石列表[]。
3.  钻石[]作为道具传递到图表和表格中。
4.  diamonds[]状态的初始和任何后续更改都会触发使用 diamonds[]重新渲染子组件。

![](img/6dbfe292ac779595fded11b78a21e90f.png)

Full results of model prices and analytic graphs + unseen tab view to tabular results

来自上一步的过滤器( [Chooser.js](https://github.com/dougfoo/diamonds/blob/master/diamond-app/src/Chooser.js) )和 [Pricer.js](https://github.com/dougfoo/diamonds/blob/master/diamond-app/src/Pricer.js) 非常相似。 [Pricer](https://github.com/dougfoo/diamonds/blob/master/diamond-app/src/Pricer.js) 是作为**函数**而不是**类**实现的，因此语法略有不同。带状态的函数是新的“钩子”,被添加到 React 中，是完整类的更好的替代品。

# 步骤 5:在 Express 中构建 REST 服务(w/ MongoDB)

Express 是 JavaScript Node.js webserver。我仍然更喜欢 Python 中的后端，但我猜这个教程是 MERN，所以我必须使用/学习 Express(这并不太难)。一个关键的优势是当您拥有相同的前端和后端库时可以重用。

同样，我不会深入安装的基础知识。我将跳转到托管我的 API 逻辑的 server.js 脚本。我需要的主要 REST 调用:

*   获得/钻石/每日-每日随机钻石
*   POST /diamonds/q/ —获取具有某些过滤标准的列表
*   POST /diamonds/q2/ —获取符合定价标准的列表
*   POST/diamonds/price——使用我的 ML 模型为钻石定价

## **首先让我们设置 MongoDB**

为 Express 设置一个 Mongoose 模型，创建一个数据库并批量加载该数据库。

1.  设置 Express 端点
2.  MongoDB 在 MLab 上免费— *FYI Azure 有 CosmoDB，它(大部分)与 MongoDB 有线兼容。CosmoDB 最便宜的计划是 24 美元/月！*

![](img/ade370281fc6198a00a00e2f08a42e42.png)

Mongo is free — vs $24/month on Azure

创建数据库后，您需要上传数据或手工输入数据。对我来说，最简单的方法是使用批量加载器脚本——要求您在 PC 上安装 MongoDB 客户端，并指向 MLAB 连接字符串。

```
mongoimport --uri mongodb://user:pass@host.mlab.com:12122/diamonds --collection myc --type csv --headerline  --file jsonfile.csv
```

上传 150k 的钻石条目速度很快，只需几分钟，并且确实使用了接近 0.5gb 的限制。全部完成！

**2。Mongoose 模型—** 创建一个脚本，定义一个与您的 MongoDB 集合的 JSON 格式相匹配的模式:

```
#diamond.model.js
let Diamond = new Schema({
  carat: {
     type: Number,
     default: -1
  },
  color: {
     type: String,
     default: 'Default'
  },
  ...
}, { collection: 'myc' });module.exports = mongoose.model('Diamond', Diamond);
```

**3。Express.js REST 端点(**[**server . js**](https://github.com/dougfoo/diamonds/blob/master/server.js)**)**都是下一步。

在那里，大部分都是样板文件。打开一个要共享的连接，设置任何备用映射(在本例中为 react path)，并监听一个端口。在这两者之间，我们为各种 GET/POST 处理程序创建路由(在下一节中定义)。

```
app.use(cors());
app.use(bodyParser.json());
mongoose.connect('mongodb://user:pass@host:port/diamonds', 
   { useNewUrlParser: true, useUnifiedTopology: true  });const connection = mongoose.connection;
connection.once('open', function() {
  console.log("MongoDB connection established successfully");
})//-- fill w/ diamondRoutes start
... see below
//-- fill endapp.use('/diamonds', diamondRoutes);// for webpack'd react front end 
app.use(express.static("./diamond-app/build")); 
app.get("*", (req, res) => {
   res.sendFile(path.resolve(__dirname, "./diamond-app", 
       "build", "index.html"));
});
app.listen(PORT, function() {
   console.log("Server is running on Port..: " + PORT);
});
```

接下来，我们填写 diamondRoutes — /daily /q /q2 和/price。/daily 很简单——只是一个返回 1 个 diamond 的随机查询(使用 diamond.model.js 中定义的 json 模式)

```
diamondRoutes.route('/daily').get(function(req, res) {
   let random = Math.floor(Math.random() * 140000);  // skip n
   Diamond.findOne().skip(random).exec(function(err, diamonds) {
      res.json(diamonds);
   });
});
```

/q 和/q2 都是类似的查询搜索，接受 POST 参数并映射到一个 Mongoose 查询——POST 输入是一个 JSON，如{carat:3，color:'D '，cut:' '，clarity:''}

```
diamondRoutes.route('/q2').post(function(req, res) {
   const qobj = req.body;
   query = Diamond.find(); 
   query = query.where('carat').
       gt(qobj.carat-0.05).lt(qobj.carat+0.05);   
   if (qobj.cut !== '') query = query.where('cut').equals(qobj.cut);
   if (qobj.color !== '') query =
      query.where('color').equals(qobj.color);
   if (qobj.clarity !== '') query =
      query.where('clarity').equals(qobj.clarity);  
   query.exec(function(err, diamonds) {   
      res.json(diamonds);
   });
});
```

![](img/852ea5186955868139b65a3e1c0b0354.png)

POSTman test

**调用/价格**—**机器学习推理调用**是通过另一个 REST API 调用 Azure ML Studio 的 farm 完成的(*现在大部分是对 Azure Flask 服务器*)。

请求链:React 客户端-> Express.js -> ML 服务->返回 Express.js ->返回 React 客户端。

我认为以块的形式显示代码会令人困惑，所以只看关键步骤，即使用“异步”调用 callSvc 函数，然后“等待”它们全部加入结果:

```
let r0 = await callSvc(p2.XGB.url, p2.XGB.token, reqJson);
let r1 = await callSvc(p2.LR.url, p2.LR.token, reqJson);
let r2 = ...
```

callSvc 只是执行一个 axios http POST 来获取 ML 预测。注意函数上的“async”定义，以及上面配对异步调用的 await 调用。

```
async function callSvc(apiurl, token, reqJson) {
    return axios.post(apiurl, reqJson, 
        { headers: {
            'Content-Type': 'application/json',
            'Authorization': token,
            }
        })
        .catch(function (error){
            console.log(error);
        });
}
```

对于前端来说，响应被合并到单个响应 JSON 中，注意 ML Studio 格式不同于我的定制 SKLearn 模型:

```
res.json([
   { price:r0.data.Results.output1.value.Values[0][6],model:'XGB' }, 
   { price:r1.data.Results.output1.value.Values[0][6],model:'LR' },
   { price:r2.data.Results.output1.value.Values[0][6],model:'NN' },
   { price: r3.data[0], model: 'ISO' },
   { price: r4.data[0], model: 'XGB2' },
   { price: r5.data[0], model: 'LR3' },
   { price: r6.data[0], model: 'RF' },
]);
```

通过 POSTMAN 查看链的外观(调用 express，它调用 ML 服务，汇总所有 7 个模型的结果):

![](img/b3e17c0d7a00316574d0d933701f7589.png)

Final test via POSTMAN->Express.js->ML Services

完整的故事请看完整的[源代码](https://github.com/dougfoo/diamonds/blob/master/server.js)。这就是大部分——我们现在已经构建了 express.js 后端！

# 步骤 6:React+Express 的 Azure 部署

部署到 Azure 站点真的很简单(当它正常工作时)——当出现构建或部署错误时会非常沮丧。祝你好运！

首先，您应该理解在开发中您可能运行 2 个服务器:

1.  反应开发引擎(默认端口 3000)
2.  快速开发服务器(默认端口 4000)

在 prod 中，您只运行一台服务器(Express ), React 的内容通过“npm build”被“捆绑”到一个 webpack 中，结果是这样的:

![](img/1aaab582cf5d82780a2df965c6ce172a.png)

webpack results

您的 express.js 有指向这个构建/静态文件夹的代码

```
app.use(express.static("./diamond-app/build"));
app.get("*", (req, res) => {
     res.sendFile(path.resolve(__dirname, "./diamond-app", 
                  "build", "index.html"));
});
```

将这个项目签入 github，你就可以开始了！(*)。git 忽略 express.js (~diamonds)和 react (~diamonds/diamond-app)级别的 node _ modules*)。

**登录 Azure 门户并创建 Web 应用**

![](img/be308aba7feb5060d6ccf02c6e430804.png)

creating an azure webapp

“testwebapp23”将变成[http://testwebapp23.azurewebsites.net](http://testwebapp23.azurewebsites.net)所以要小心命名，它**以后不能更改**(如果你拥有一个域，你可以在外部给它起个 DNS 别名)。选择最便宜的尺寸，B1 将花费你大约 15 美元一个月！

我会检查 Azure Web App 的配置面板，确认一些事情:

![](img/1613834d37211b26a7bd0796488b3b5c.png)

double check the config

接下来，您需要在**Azure devo PS**([https://dev.azure.com](https://dev.azure.com))中创建一个项目来托管构建环境。然后，转到项目设置(左下角)并将其链接到您的 GitHub repo，并启用管道:

![](img/18346ce99930dfa86091c9adfe69846c.png)

管道将启用您的 CI/CD 设置，因此对 GitHub 的签入将触发构建和部署。有几种方法来设置它，每种方法都有其缺陷。奇怪但半简单的方法是跳回 Azure Portal，用他们的 webapp-> **部署中心**结束。

![](img/614e1e31d6887449683e44c5c6cec062.png)

Select GitHub — then Azure Pipelines

你可以选择 Kudu App Build 或者 Azure Pipelines。Pipelines 是最新的 Azure CI/CD 工具集(取代了 TFS)。两者都有问题..但是我发现对于 node.js 管道部署效果更好(对于 Python 我更喜欢 Kudu)。

![](img/5f202c600d0eed7fa6bb3ac11e7bae93.png)

完成，这将创建您的第一个构建和部署。如果你幸运的话，它可以工作，你可以像这样在线查看你的应用程序:

![](img/66c46b619e8ae09e91fdcbec1df5b96c.png)

It works!

跳回 Azure DevOps，你会看到它生成了一个管道(构建)和发布(部署):

![](img/18878e798591d609ef054d7df8ae9d7b.png)

Code successfully built in Azure Pipelines

如果通过，则进入发布(部署):

![](img/f69d120ea38cfe225bd70213801c6db4.png)

您可以定制每个步骤，并添加一些漂亮的东西，如前/后脚本或如下所示的批准流程，以在您的环境中传播更改。一个变更触发了一个预生产(演示)构建，然后一旦我验证了构建，我必须手动批准发布到生产(diamonds.foostack.ai)。

![](img/6317d13d5b20f3a722ec03c7629fb81c.png)

# **第七步:技巧和烦恼**

我学到并想提醒你的最重要的事情:

*   **Azure Insights** 非常适合跟踪使用统计数据(设置起来比你想象的要困难，你必须将代码与你的每个组件挂钩，但之后你会得到一些很好的遥测数据)

![](img/5109750cfd7b2544080a054af5fe944a.png)

*   Azure 正在发展(正在崩溃),所以准备好面对一些麻烦，尤其是机器学习和管道等新事物(linux 上的 node.js 版本突然崩溃..几个星期以来，Azure 团队一直在努力清理这场灾难:

![](img/6c24ba7fdcb1237bdd7f8041d17478a2.png)

*   **Scrum Boards** 和票证管理可以在 Azure DevOps 中完成，并与 GitHub 集成。它很新，但对于一个小项目来说还不错。

![](img/6df2fe76b9bd6f19cb5a6b4f2c17e161.png)

*   **成本管理** —密切跟踪。每个机器学习模型的成本有点疯狂(每个模型容器/天 2 美元，仅 12 天就花了我 100 多美元)。这促使我迁移到 Flask。在另一个项目中，SQL Warehouse 用了 2 天就超过了 100 美元！小心，设置成本警报！

![](img/00709f49bed258e0f5117c258992d332.png)

*   [**Python Flask Deploy**](https://github.com/dougfoo/flaskml)**——因为上面的成本问题，我晚加了这个。我使用传统的来自 WebApp- > Deployment 的 Kudu build，而不是 Azure Pipelines，因为我一直使用 Azure Pipelines 时出现了一些错误。**

**![](img/2f9faf264e93339f0e5af8f58b097a32.png)**

**Python Flask using older Kudu deployment (works !)**

*   ****域名**设置真的很简单——Azure 不这么做，去[namecheap.com](http://namecheap.com)——然后在 Azure 上链接就行了。非常简单的说明。**

**![](img/33eb0598ef72d04ad7d6cee4c4e94d15.png)**

**编码快乐！在线阅读 live[http://diamonds . foo stack . ai](http://diamonds.foostack.ai)网站、 [github 项目](https://github.com/dougfoo/diamonds)或 [Jupyter 笔记本](https://nbviewer.jupyter.org/github/dougfoo/machineLearning/blob/master/diamonds/Diamond-Analysis-4.ipynb)。**