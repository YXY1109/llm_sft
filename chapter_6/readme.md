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
python -m vllm.entrypoints.openai.api_server --model /root/autodl-tmp/llm_sft/chapter_5/models/Qwen/Qwen3-8B --served-model-name qwen3-8b --trust_remote_code --gpu_memory_utilization 0.95 --max-num-seqs 256 --max_model_len 4096 --port 8881
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
    "max_tokens": 500,
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
- python Qwen2.5-0.5B.py
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
- python Qwen8-8B.py
- 完整结果：./qwen3-8b目录

**评估结果**

| Model    | Dataset | Metric          | Subset        | Num | Score | Cat.0   |
|----------|---------|-----------------|---------------|-----|-------|---------|
| qwen3-8b | arc     | AverageAccuracy | ARC-Easy      | 50  | 0.94  | default |
| qwen3-8b | arc     | AverageAccuracy | ARC-Challenge | 50  | 0.94  | default |
| qwen3-8b | arc     | AverageAccuracy | OVERALL       | 100 | 0.94  | -       |
| qwen3-8b | gsm8k   | AverageAccuracy | main          | 50  | 0.94  | default |

## 四，评估模型(详细使用)
