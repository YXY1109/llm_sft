from modelscope import snapshot_download

dp_model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', cache_dir="./models")
print(f"deepseek模型目录：{dp_model_dir}")
# qwen_model_dir = snapshot_download('Qwen/Qwen3-8B', cache_dir="./models")
# print(f"qwen3模型目录：{qwen_model_dir}")
print("下载完成")
