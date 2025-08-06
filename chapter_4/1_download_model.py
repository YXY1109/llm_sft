from modelscope import snapshot_download

model_dir = snapshot_download('Qwen/Qwen2.5-3B-Instruct', cache_dir="./models")
print(model_dir)
print("下载完成")
