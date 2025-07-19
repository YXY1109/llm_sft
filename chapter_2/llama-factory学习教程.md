# 启动服务

> https://github.com/hiyouga/LLaMA-Factory

```
安装torch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

windows执行命令：
$env:USE_MODELSCOPE_HUB=1
llamafactory-cli webui
```

## 计算token数

> https://tiktokenizer.vercel.app/?model=Qwen%2FQwen2.5-72B

## 数据来源

> https://github.com/ConardLi/easy-dataset
> 数据文件：/Users/cj/PycharmProjects/LLM_SFT/chapter_2/data_files/easy-data

## 分析数据脚本大小的分布

```
用于设置max_length_tokens
python .\scripts\stat_utils\length_cdf.py --help
python .\scripts\stat_utils\length_cdf.py --model_name_or_path D:\PycharmProjects\LLaMA-Factory\saves\Qwen3-0.6B-Instruct\Qwen3-0.6B-Instruct-yxy --dataset identity --template qwen --interval 10
    
```

## 模型上传huggingface

```
执行upload_model.py脚本
```

## 模型转为gguf

```
转换教程：
https://www.novishare.site/notes/fine-tuning/fk4ywqye/
https://github.com/ggml-org/llama.cpp/discussions/2948

clone llama.cpp源码
安装torch：pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
安装依赖：pip install --editable .

choose from 'f32', 'f16', 'bf16', 'q8_0', 'tq1_0', 'tq2_0', 'auto'

python convert_hf_to_gguf.py D:\PycharmProjects\LLaMA-Factory\export_model --outtype q8_0 --verbose --outfile D:\PycharmProjects\LLaMA-Factory\export_model\Qwen3-0.6B-Instruct-yxy_q8_0.gguf
```

## ollama推送

```
注意：目前ollama对qwen3支持还不够好。https://blog.csdn.net/weixin_44354960/article/details/147950932

ollama create yxy1109/Qwen3-0.6B-Instruct-yxy -f D:\PycharmProjects\LLaMA-Factory\yxy\Modelfile
ollama push yxy1109/Qwen3-0.6B-Instruct-yxy
```

## 自我认知微调

问题：

```
你是谁
你叫什么名字
你是谁开发的
```

## 格力2023年年报微调

```
财务报告报出的时间是什么时候? 
答案：2024 年 4 月 29 日

公司治理的基本状况

```
