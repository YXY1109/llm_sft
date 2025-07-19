# 模型下载
from modelscope import snapshot_download

model_dir = snapshot_download('Qwen/Qwen3-1.7B', cache_dir="./models")
print(model_dir)
print("下载完成")
