# LLM模型训练学习记录

## chapter_1

从零开始训练模型

- 使用minimind在autodl上训练模型，包括预训练模型和微调模型

## chapter_2

构造数据集

- 使用easy-data构造数据集
- 使用llama-factory微调模型
- 然后将模型合并，上传到huggingface
- 使用llama.cpp将模型转为gguf，上传到ollama

## chapter_3

微调模型，手写实现

- 微调模型：Qwen3微调实战：医疗R1推理风格聊天
- 微调数据：[delicate_medical_r1_data](https://modelscope.cn/datasets/krisfu/delicate_medical_r1_data)
- 微调框架：基于transformers，手写实现

## chapter_4

微调模型，使用LLama-Factory

- 微调模型：Qwen2.5-3B-Instruct
- 微调数据：[财报数据](https://github.com/llm-factory/FinancialData-SecondQuarter-2024)
- 微调框架：LLama-Factory

## chapter_5

微调模型，使用Unsloth

- 微调模型：Qwen3-8B和DeepSeek-R1-Distill-Qwen-7B
- 微调数据：[medical-o1-reasoning-SFT](https://modelscope.cn/datasets/AI-ModelScope/medical-o1-reasoning-SFT)
- 微调框架：Unsloth

## chapter_6

评估模型，使用evalscope

- 使用[evalscope](https://github.com/modelscope/evalscope)评估模型

## chapter_7

收集医疗数据集，清洗数据

## chapter_8

transformers学习