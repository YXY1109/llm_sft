# 评估模型

## 一，部署模型

- 进入[Autodl](https://autodl.com/home)后台
- 安装依赖包

```
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install vllm==0.10.0
```

- 推理服务

```
原始模型推理服务
python -m vllm.entrypoints.openai.api_server --model /root/autodl-tmp/llm_sft/chapter_5/models/Qwen/Qwen3-8B --served-model-name qwen3-8b --trust_remote_code --gpu_memory_utilization 0.95 --max-num-seqs 256 --max_model_len 4096 --port 8881

微调模型推理服务
python -m vllm.entrypoints.openai.api_server --model /root/autodl-tmp/llm_sft/chapter_5/QWen3-8B-Medical-CoT --served-model-name qwen3-8b-lora --trust_remote_code --gpu_memory_utilization 0.95 --max-num-seqs 256 --max_model_len 4096 --port 8882
```

- windows系统下载：audl隧道工具，配置隧道端口
- curl命令测试

```
curl --location 'http://127.0.0.1:8881/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
    "model": "qwen3-8b",
    "messages": [
      {"role": "user", "content": "你是什么模型"}
    ],
    "temperature": 0.7,
    "max_tokens": 50,
    "stream":false
  }'
```

## 二，部署evalscope

- 本地下载源码：git clone git@github.com:modelscope/evalscope.git
- 安装依赖包

```
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

## 三，评估模型(简单使用)

### 3.1 评估Qwen2.5-0.5B-Instruct

- 本地模型没有，会从modelscope下载
- python 1_qwen2.5_0.5b.py
- 完整结果：./qwen2.5-0.5b目录

**评估结果**

| Model                 | Dataset | Metric          | Subset        | Num | Score | Cat.0   |
|-----------------------|---------|-----------------|---------------|-----|-------|---------|
| Qwen2.5-0.5B-Instruct | arc     | AverageAccuracy | ARC-Easy      | 50  | 0.68  | default |
| Qwen2.5-0.5B-Instruct | arc     | AverageAccuracy | ARC-Challenge | 50  | 0.38  | default |
| Qwen2.5-0.5B-Instruct | arc     | AverageAccuracy | OVERALL       | 100 | 0.53  | -       |
| Qwen2.5-0.5B-Instruct | gsm8k   | AverageAccuracy | main          | 50  | 0.42  | default |

### 3.2 评估Qwen3-8B

- 使用OpenAi的api
- python 2_qwen3_8b.py
- 完整结果：./qwen3-8b目录

**评估结果**

| Model    | Dataset | Metric          | Subset        | Num | Score | Cat.0   |
|----------|---------|-----------------|---------------|-----|-------|---------|
| qwen3-8b | arc     | AverageAccuracy | ARC-Easy      | 50  | 0.94  | default |
| qwen3-8b | arc     | AverageAccuracy | ARC-Challenge | 50  | 0.94  | default |
| qwen3-8b | arc     | AverageAccuracy | OVERALL       | 100 | 0.94  | -       |
| qwen3-8b | gsm8k   | AverageAccuracy | main          | 50  | 0.94  | default |

## 四，评估模型(详细使用)

## 4.1 创建自定义数据集

> [微调数据集](https://modelscope.cn/datasets/AI-ModelScope/medical-o1-reasoning-SFT/files)
>
> 微调的数据集来自于：medical_o1_sft_Chinese.json这个文件

> [评测数据集](https://modelscope.cn/datasets/krisfu/delicate_medical_r1_data/files)
>
> 部分数据构建数据集。[构建教程](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html)

## 4.2 评估模型

- 构建原始模型的推理服务
- 构建数据集：python 3.0_build_data.py
- 执行rouge和bleu评估：python 3.1_rouge_bleu.py
- 执行LLM评估：python 3.2_llm_eval.py

## 4.3 评估模型

- 构建微调模型的推理服务
- 注意模型名称和端口号

## 4.4 评估结果

outputs目录下查看结果，**简单分析评估结果：**

- 基于rouge和bleu的微调后好些
- 基于ll_eval的原始模型好些

