# 如何在分析中避开 SQL 中间人

> 原文：<https://towardsdatascience.com/how-to-cut-out-the-sql-middle-person-in-analytics-9687d640cecc?source=collection_archive---------12----------------------->

## 不断地为您的客户手动执行 SQL 查询？这里有一个让他们自助的方法。

我目前的人生目标之一是帮助数据分析师削减他们工作中无聊、麻木的方面，以便他们可以专注于更有趣、有用和酷的东西。

我在分析团队中看到的一种常见情况是，分析师扮演了一个不必要且高度重复的中间人角色，即业务用户反复需要某些特定数据，而由分析师手动执行这些重复的 SQL 查询来提取数据并将其发送给客户端。相同或非常相似的查询，一次又一次。不好玩。实际上在今天的技术下完全没有必要。

我爱 R 生态系统。我喜欢它的原因之一是因为它可以让你轻松地做一些神奇的事情，将数据源与基于 web 的交互联系起来。在这篇文章中，我将向您展示如何使用 R Shiny 建立一个简单的 web 应用程序，它将允许您的非技术专业客户从 SQL 数据库中选择他们想要的原始数据，并将其提取和下载到 Excel 电子表格中。这是去掉中间人的安全方法，因为您可以控制谁可以访问 Web 应用程序，以及允许它从数据库中检索哪些数据。

我将在这里使用一个简单的场景来说明这些概念，但是这个想法可以扩展到能够向您的客户端提供大量可下载的预处理数据。

## 你需要什么

1.  r 安装了`shiny`、`dplyr`、`openxlsx`、`rmarkdown`和`odbc`包。
2.  您希望从中检索特定数据的 SQL 数据库。在本文中，我将假设它是一个 Oracle SQL 数据库。
3.  RStudio Connect 文档/应用托管平台，或至少访问一个闪亮的服务器。

## 我们的简单示例场景

让我们假设你是一家连锁兽医诊所的数据分析师。每天您都会收到一些关于预约和预约取消的数据请求。我们会问两个问题:在这几年中，我们为每种动物预约了多少次？其中有百分之多少被取消了？您已经厌倦了一次又一次地使用相同的 SQL 查询来回答这些问题。

您的约会数据库表称为`APPTS_DB`，它有许多列，特别是`DATE`(格式为`DD-MMM-YYYY`，例如`01-MAR-2017`)、`ANIMAL_TYPE`(例如`Cat`、`Dog`)、`CANCEL_FLAG`(如果约会被取消，则为二进制`1`，否则为`0`)。

我假设你知道闪亮应用的结构。在这篇文章中，我将把这个闪亮的应用写成一个交互式 R markdown `.Rmd`文档。有关这方面的更多信息，请参见[此处](https://shiny.rstudio.com/articles/rmarkdown.html)。

## 步骤 1:将您的 SQL 查询转换成 R 中的函数

将常用的 SQL 查询转换成 R 函数确实是一个好主意，即使您在本文中没有做任何其他事情。它允许您在 R 会话中用一个命令执行常见的查询。

为了将这一点构建到您的应用程序中，这一步可以在一个单独的 R 脚本中完成，您可以将其加载到您闪亮的应用程序中，或者它可以作为应用程序本身的代码块来完成。我们假设是后者。

首先，我们建立一个到数据库的连接。如果您打算将代码存储在 Github 或其他任何可能被他人看到的地方，我建议将您的凭证放在系统环境中。

```
``` {r db_connect}db_conn <- odbc::dbConnect(odbc::odbc(), 
                          dsn = Sys.getenv("DSN"), # <- database
                          uid = Sys.getenv("UID"), # <- user id
                          pwd = Sys.getenv("PASS")) # <- password```
```

首先，我们用占位符编写 SQL 查询，然后使用`gsub()` 用函数参数替换占位符，然后运行查询。我们希望设置它来提取从指定的起始年到结束年的时间段内按动物类型的约会总数，并计算取消率。理想情况下，我会建议您使用`dbplyr`以整洁的形式完成这个查询，但是我在这里将使用原始 SQL 来完成，因为大多数读者对此都很熟悉。

```
``` {r query_fn}# create query functionappt_query <- function (start_yr, end_yr) {# write query with placeholders qry <- "SELECT ANIMAL_TYPE, COUNT(*) NUM_APPTS, 
          SUM(CANCEL_FLAG)/COUNT(*) PERC_CANCELLED
          FROM APPTS_DB
          WHERE EXTRACT(YEAR FROM DATE) BETWEEN start_year AND end_year          
          GROUP BY ANIMAL_TYPE
          ORDER BY ANIMAL_TYPE"# replace placeholders with function arguments qry <- gsub("start_year", start_yr, qry)

  qry <- gsub("end_year", end_yr, qry)# execute query odbc::dbGetQuery(db_conn, qry)} ```
```

这非常强大，因为它允许使用简单的命令如`appt_query(2015, 2017)`获取信息。您也可以使用一个简单的`paste()`函数将 SQL 查询粘贴在一起，但是我发现使用`gsub()`更容易管理和检测错误。

## 步骤 2:设置闪亮的应用程序输入和下载按钮

现在，我们为最终用户设置了一些简单的输入，以及一个下载按钮，他们将按下该按钮来执行查询并下载他们需要的数据。

在这个简单的例子中，只需要用户输入起始年份和结束年份。我们将从 2010 年开始我们的年度期权，到今年结束

```
``` {r user_input}# get current yearcurr_year <- format(Sys.Date(), "%Y")# set up panel to select start and end yearsmainPanel(
  HTML("<br>"),
  selectInput("start_yr", "Select start year:",
              choices = c(2010:curr_year)),
  selectInput("end_yr", "Select end year:", 
              choices = c(2010:curr_year)),
  HTML("<br>"),
  uiOutput("downloadExcel")
)# generate download button confirming the input selectionoutput$downloadExcel <- renderUI(
  downloadButton("downloadData", paste("Download Data for", input$start_yr, "to", input$end_yr, "in Excel"))
)```
```

这将创建一个带有下载按钮的用户输入面板，确认所选择的年份。这个 download 按钮将链接到一个名为`downloadData`的服务器进程，在这里我们将执行查询并写入一个 Excel 文件供用户下载。

## 步骤 3:编写后台代码来执行查询和下载数据

现在我们编写服务器进程来执行查询和下载数据。这变得更简单了，因为我们设置了一个简洁的函数来执行查询(参见步骤 1)。我们还将建立一个整洁的进度条，以便用户可以看到该进程是如何进展的。对于一个可以立即完成的简单查询来说，这是不必要的，但是如果您正在使用许多不同的查询构建一个多选项卡的电子表格，那么它是必不可少的，以便让用户确信正在发生一些事情。

```
``` {r query_and_download}# use downloadHandleroutput$downloadData <- downloadHandler(# give download file a name   filename = function() {
    paste0("animal_appt_data_", input$start_yr, "_", input$end_yr, ".xlsx")
    },# download and populate excel file content = function(file) {# create progress bar for user with first step

    shiny::withProgress(
          min = 0,
          max = 1,
          value = 0,
          {
            shiny::incProgress(
              amount = 1/2,
              message = "Retrieving data from database..."
              )

# get data using function data <- appt_query(input$start_yr, input$end_yr)# set up an excel file in a tab called "Appointment Data" shiny::incProgress(
          amount = 1/2,
          message = "Writing to Excel..."
        ) wb <- openxlsx::createWorkbook()
  openxlsx::addWorksheet(wb, "Appointment Data") # design a nice header style so the results look professional hs1 <- openxlsx::createStyle(fontColour = "#ffffff", 
                               fgFill = "#4F80BD",
                               halign = "left", 
                               valign = "center", 
                               textDecoration = "bold",
                               border = "TopBottomLeftRight", 
                               wrapText = TRUE)# write the results into the "Appointment Data" tab with a nice border openxlsx::writeData(wb, "Appointment Data", x = data,
                      startRow = 1, startCol = 1, 
                      borders = "surrounding",
                      headerStyle = hs1)# save Excel file and send to download openxlsx::saveWorkbook(wb, file, overwrite = TRUE) } # <- end progress bar 
) # <- end withProgress})  # <- close content wrapper and downloadHandler function```
```

## 步骤 4:向客户端提供访问权限

使用这种方法，客户端只能访问您为其设计的数据。如果您使用 RStudio Connect 共享平台，您可以将其发布为应用程序，并仅向特定个人或群组提供访问权限，确保只有授权的客户端才能访问。如果您使用 Shiny Server Pro，请从用户文档中寻求如何控制用户访问的建议。

## 结论和延伸

如果你是一个流利的闪亮的用户，你可能会立即掌握这里发生了什么。如果没有，你可能需要更多的培训，在你接受这个之前。这种方法可以扩展到非常强大。仪表板可以使用`ggplot2`或`plotly`来设计，而不是——或者除了——Excel 下载。像`async`、`future`和`promises`这样的包可以用于使用异步编程的多个并发用户的伸缩(更多信息见[这里](https://blog.rstudio.com/2018/06/26/shiny-1-1-0/))。在最高级的情况下，可以使用这里介绍的初始概念在 Shiny 上构建和托管一个完整的灵活的商业智能平台。

最初我是一名纯粹的数学家，后来我成为了一名心理计量学家和数据科学家。我热衷于将所有这些学科的严谨性应用到复杂的人的问题上。我也是一个编码极客和日本 RPG 的超级粉丝。在[*LinkedIn*](https://www.linkedin.com/in/keith-mcnulty/)*或*[*Twitter*](https://twitter.com/dr_keithmcnulty)*上找我。*

![](img/3c94322805b1e2fa4ee0c52ebedc2a02.png)