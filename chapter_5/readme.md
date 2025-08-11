# LoRA微调指南，打造你自己的医疗专家

## [DeepSeek-R1-Distill和Qwen3-8B模型](https://docs.alayanew.com/docs/documents/bestPractice/bigModel/deepSeekR1DistillSft/

## 操作流程

- 登录AutoDL后台，选择GPU服务器，我一般选择3090，因为比较便宜，1.56元/小时
- 克隆代码：git clone 代码
- 安装依赖包：pip install -r requirements.txt
- 下载模型：python 1_download_model.py
- 微调模型：python 2.*.py
- 推理原始和微调后模型：python 3.*.py

## Swanlab

- [qwen3-8b微调结果](https://swanlab.cn/@YXY1109/chapter_5/runs/n17u1fsglugt1s0ba2lq8/chart)
- [deepseek微调结果](https://swanlab.cn/@YXY1109/chapter_5/runs/p6zgm0ljxdevk91kbohob/chart)
