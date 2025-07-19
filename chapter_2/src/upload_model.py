from huggingface_hub import login, HfApi
from transformers import AutoModelForSequenceClassification, AutoTokenizer

"""
访问 Hugging Face 官网 并登录账户:https://huggingface.co/
点击右上角头像 → Settings → Access Tokens。
生成一个具有write权限的新令牌（记录该令牌，后续会用到）。
"""
login(token="")
print("登录成功！")

# 加载本地微调好的模型和分词器
model_dir = r"D:\PycharmProjects\LLaMA-Factory\export_model"
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
print("模型加载成功！")

# 创建模型仓库（可选，若已手动创建可跳过）
# 手动创建地址：https://huggingface.co/new
api = HfApi()
repo_id = "yangdaye/Qwen3-0.6B-Instruct-YXY"  # 替换为你的用户名和模型名称
api.create_repo(repo_id=repo_id, exist_ok=True)
print("模型仓库创建成功！")

# 推送模型和分词器到Hub
model.push_to_hub(
    repo_id=repo_id,
    commit_message="上传使用llama-factory微调的模型qwen3-0.6b-instruct模型",
    private=False,
    tags=["text-generation"]
)

tokenizer.push_to_hub(repo_id=repo_id)

print(f"模型已成功推送到：https://huggingface.co/{repo_id}")
