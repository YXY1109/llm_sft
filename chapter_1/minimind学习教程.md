# Minimind模型训练

> https://github.com/jingyaogong/minimind

## 预训练和SFT微调，基于AutoDL

## 拉取github代码1

```
github设置：settings/Developer Settings/Personal access tokens/Fine-grained tokens

服务器执行命令：
git clone https://oauth2:{申请的token}@github.com/jingyaogong/minimind.git

```

## 拉取github代码2

```
github设置：settings/SSH and GPG keys/SSH key

ssh-keygen -t rsa -b 4096 -C "123456@qq.com"
cat ~/.ssh/id_rsa.pub
git clone git@github.com:YXY1109/llm_sft.git
```

## 查看GPU

```
pip install nvitop
方便查看gpu使用情况：
nvitop
```

# 安装报错

```
以下是autodl，ubuntu2204自带的，不需要安装，删除掉
torch==2.3.0
torchvision==0.18.0

Failed to build tiktoken无法安装：
将requirements.txt中的包删除：
tiktoken==0.5.1

pip download tiktoken --only-binary=:all:
pip install tiktoken-*.whl

```