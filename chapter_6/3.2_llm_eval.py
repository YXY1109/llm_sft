import os

from dotenv import load_dotenv
from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType, JudgeStrategy

"""
https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html
方法2. 基于LLM的评测
"""

load_dotenv()  # 加载.env文件

task_cfg = TaskConfig(
    model='qwen3-8b',  # 原模型：qwen3-8b；微调模型：qwen3-8b-lora
    api_url='http://127.0.0.1:8881/v1',  # 原模型：8881；微调模型：8882
    api_key="none",
    eval_type=EvalType.SERVICE,
    datasets=[
        'general_qa',
    ],
    dataset_args={
        'general_qa': {
            "local_path": "D:\PycharmProjects\llm_sft\chapter_6\json_data",  # 自定义数据集路径
            "subset_list": [
                # 评测数据集名称，上述 *.jsonl 中的 *，可配置多个子数据集
                "r1_data_example_100"
            ]
        }
    },
    # judge 相关参数
    judge_model_args={
        'model_id': 'qwen2.5-72b-instruct',
        'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key': os.getenv('DASHSCOPE_API_KEY'),
        'generation_config': {
            'temperature': 0.0,
            'max_tokens': 4096
        },
        # 根据参考答案和模型输出，判断模型输出是否正确
        'score_type': 'pattern',
    },
    # judge 并发数
    judge_worker_num=5,
    # 使用 LLM 进行评测
    judge_strategy=JudgeStrategy.LLM,
)

run_task(task_cfg=task_cfg)
