import os

from dotenv import load_dotenv
from huggingface_hub import HfApi
from modelscope.hub.api import HubApi


def load_environment_variables():
    """加载.env文件中的环境变量"""
    load_dotenv()  # 加载.env文件

    # 获取环境变量
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    ms_token = os.getenv("MODELSCOPE_TOKEN")
    model_path = os.getenv("MODEL_PATH")
    hf_repo_name = os.getenv("HUGGINGFACE_REPO_NAME")
    ms_repo_name = os.getenv("MODELSCOPE_REPO_NAME")

    # 验证必要的环境变量是否存在
    required_vars = {
        "HUGGINGFACE_TOKEN": hf_token,
        "MODELSCOPE_TOKEN": ms_token,
        "MODEL_PATH": model_path,
        "HUGGINGFACE_REPO_NAME": hf_repo_name,
        "MODELSCOPE_REPO_NAME": ms_repo_name
    }

    for var_name, var_value in required_vars.items():
        if not var_value:
            raise ValueError(f"环境变量 {var_name} 未设置，请检查.env文件")

    return {
        "hf_token": hf_token,
        "ms_token": ms_token,
        "model_path": model_path,
        "hf_repo_name": hf_repo_name,
        "ms_repo_name": ms_repo_name
    }


def upload_to_huggingface(model_path, repo_name, token):
    """
    将模型上传到Hugging Face Hub。在autodl上网络不同，所以失败
    :param model_path: 模型的目录
    :param repo_name: 仓库名称
    :param token:
    :return:
    """
    try:
        print(f"开始上传模型到Hugging Face仓库: {repo_name}")
        api = HfApi()
        api.create_repo(repo_id=repo_name, exist_ok=True, token=token)
        print(f"成功创建模型仓库: https://huggingface.co/{repo_name}")

        # 上传模型
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            repo_type="model",
            token=token
        )
        print(f"模型成功上传到Hugging Face: https://huggingface.co/{repo_name}")
        return True
    except Exception as e:
        print(f"上传到Hugging Face失败: {str(e)}")
        return False


def upload_to_modelscope(model_path, repo_name, token):
    """将模型上传到ModelScope"""
    try:
        print("开始登录ModelScope...")
        api = HubApi()
        api.login(token)

        # 创建模型仓库（如果不存在）
        api.create_repo(repo_id=repo_name, exist_ok=True, token=token)
        print(f"成功创建模型仓库: https://modelscope.cn/models/{repo_name}")

        # 上传模型
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            repo_type="model",
            token=token
        )

        print(f"模型成功上传到ModelScope: https://modelscope.cn/models/{repo_name}")
        return True
    except Exception as e:
        print(f"上传到ModelScope失败: {str(e)}")
        return False


def main():
    """主函数"""
    try:
        # 加载环境变量
        config = load_environment_variables()

        # 上传到Hugging Face
        hf_success = upload_to_huggingface(
            model_path=config["model_path"],
            repo_name=config["hf_repo_name"],
            token=config["hf_token"]
        )

        # 上传到ModelScope
        ms_success = upload_to_modelscope(
            model_path=config["model_path"],
            repo_name=config["ms_repo_name"],
            token=config["ms_token"]
        )

        if hf_success and ms_success:
            print("所有模型上传成功！")
        else:
            print("部分模型上传失败，请检查错误信息")

    except Exception as e:
        print(f"程序执行出错: {str(e)}")


if __name__ == "__main__":
    main()
