from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType

task_cfg = TaskConfig(
    model='qwen3-8b',
    api_url='http://127.0.0.1:8881/v1',
    api_key="none",
    eval_type=EvalType.SERVICE,
    datasets=[
        'gsm8k',
        'arc',
    ],
    limit=50
)
run_task(task_cfg=task_cfg)
