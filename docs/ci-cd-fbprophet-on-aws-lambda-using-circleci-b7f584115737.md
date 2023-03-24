# CI/CD fbprophet 到 AWS Lambda

> 原文：<https://towardsdatascience.com/ci-cd-fbprophet-on-aws-lambda-using-circleci-b7f584115737?source=collection_archive---------21----------------------->

## 使用 CircleCI 将预测整合到您的开发流程中

![](img/426ac3eb7d8dd055c65eb75851954455.png)

Greg Studio / Unsplash

大约一年前，我试图弄清楚如何让 fbprophet 在 AWS Lambda 上进行预测工作，并最终让它工作起来。从那时起，对无服务器预测的需求成熟了，越来越多的人对无缝技术感兴趣，导致了一些非常有用的东西，如由[马克·梅斯](https://medium.com/u/9da2f3392040?source=post_page-----b7f584115737--------------------------------)为 AWS Lambda 开发的 fbprophet [汇编器](https://medium.com/@marc.a.metz/docker-run-rm-it-v-pwd-var-task-lambci-lambda-build-python3-7-bash-c7d53f3b7eb2)。

让软件包正常工作的一个自然发展是让它成为你的 CI/CD 过程的一部分。我需要经常部署我的预测代码——将预测微服务集成到其他应用程序中——手动生成和上传 zip 不是一个选项。我要分享一下我是如何为 CircleCI 解决的。希望您会发现它对其他 CI/CD 解决方案有用并可重用。

一致部署的基础是预装在 AWS Lambda Python 3.6 环境中的这个 [Docker 映像](https://hub.docker.com/r/amacenov/aws-lambda-fbprophet)。它有完整的、未编辑的 fbprophet，因此可以在 CI/CD 端进行定制。例如，您可能希望也可能不希望从 Docker 中卸载 matplotlib。

你可以随意使用这张图片。下面是它的文档(根据马克·梅茨的而建)，这里是整个回购。

```
# DockerfileFROM lambci/lambda:build-python3.6ENV *VIRTUAL_ENV*=venv
ENV *PATH* $*VIRTUAL_ENV*/bin:$*PATH*RUN python3 -m venv $*VIRTUAL_ENV* RUN . $*VIRTUAL_ENV*/bin/activateCOPY requirements.txt .
RUN pip install — upgrade pip
RUN pip install -r requirements.txt# requirements.txt
pystan==2.18
fbprophet
```

下面是我用的 CircleCI config.yml 代码。它利用 Docker 中设置的 virtualenv 的路径来准备 fbprophet 进行部署。

```
# config.yml# assumes the repo structure like this:
# repo_name
#   main_repo_folder
#      ...
#   name_of_lambda_handler.py
#   requirements.txtversion: 2
jobs:
  build:
    docker:
      - image: amacenov/aws-lambda-fbprophet
        environment:
          AWS_ACCESS_KEY_ID: 'xxx'
          AWS_SECRET_ACCESS_KEY: 'xxx'
          AWS_DEFAULT_REGION: 'xxx'

    steps:
      - checkout

      - restore_cache:
          keys:
            - v1-dependencies- *#{{ checksum "requirements.txt" }}* - run:
          name: install dependencies
          command: |
            VIRTUAL_ENV=/var/task/$VIRTUAL_ENV  # path was set up in the Docker image
            source $VIRTUAL_ENV/bin/activate
            pip install -r requirements.txt

      - run:
          name: make prophet small enough for lambda
          command: |
            VIRTUAL_ENV=/var/task/$VIRTUAL_ENV
            source $VIRTUAL_ENV/bin/activate
            pip uninstall -y matplotlib
            find "$VIRTUAL_ENV/lib/python3.6/site-packages" -name "test" | xargs rm -rf
            find "$VIRTUAL_ENV/lib/python3.6/site-packages" -name "tests" | xargs rm -rf
            rm -rf "$VIRTUAL_ENV/lib/python3.6/site-packages/pystan/stan/src"
            rm -rf "$VIRTUAL_ENV/lib/python3.6/site-packages/pystan/stan/lib/stan_math/lib"

      - run:
          name: prepare zip
          command: |
            VIRTUAL_ENV=/var/task/$VIRTUAL_ENV
            pushd $VIRTUAL_ENV/lib/python3.6/site-packages/
            zip -9qr ~/repo/name_of_the_zip.zip *
            popd
            zip -9r name_of_the_zip.zip main_repo_folder/
            zip -9 name_of_the_zip.zip name_of_lambda_handler.py- save_cache:
          paths:
            - ./cache
          key: v1-dependencies- *#{{ checksum "requirements.txt" }}* - run:
          name: push to s3
          command: aws s3 cp name_of_the_zip.zip s3://your_s3_bucket_name/name_of_the_zip.zip

      - run:
          name: push function code
          command: >
            aws lambda update-function-code --function-name your_aws_lambda_function_arn
            --s3-bucket your_s3_bucket_name
            --s3-key name_of_the_zip.zip

workflows:
  version: 2
  deploy:
    jobs:
      - build:
          filters:
            branches:
              only: [master]
```

下一次将分享更多关于我如何大规模使用 fbprohet 的信息。

干杯！