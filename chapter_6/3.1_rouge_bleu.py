from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType

"""
https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html
方法1. 基于ROUGE和BLEU评测
"""
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
)

run_task(task_cfg=task_cfg)
