from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType

task_cfg = TaskConfig(model="Qwen/Qwen2.5-0.5B-Instruct", datasets=['gsm8k', 'arc'], limit=50)
run_task(task_cfg=task_cfg)
