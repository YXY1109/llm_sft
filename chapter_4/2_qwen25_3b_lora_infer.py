from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ========== 用户只需改这里的 2 个路径 ==========
BASE_MODEL_PATH = r"D:\PycharmProjects\llm_sft\chapter_4\models\Qwen\Qwen2.5-3B-Instruct"
LORA_ADAPTER_PATH = r"D:\PycharmProjects\llm_sft\chapter_4\models_lora\train_2025-07-31-07-23-06"


# ============================================


# ---------- 自动选择最优设备 ----------
def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = auto_device()
TORCH_DTYPE = torch.float16  # 显存够可换成 torch.bfloat16


# ------------------------------------


def load_model_and_tokenizer():
    """加载 tokenizer、基座模型、LoRA adapter，并返回已 eval() 的模型"""
    print(f"[INFO] 自动选择设备: {DEVICE}")

    print("[INFO] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        padding_side="left"  # 批量推理时左侧填充
    )
    # 防止 pad_token 缺失导致 batch 推理失败
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] 加载基座模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=TORCH_DTYPE,
        device_map=DEVICE,
        trust_remote_code=True
    )

    print("[INFO] 加载 LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    model.eval()  # 关闭 dropout 等随机层
    return model, tokenizer


def chat_generate(
        model,
        tokenizer,
        questions: list[str],
        *,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9
) -> list[str]:
    """
    批量推理函数
    :param model: PeftModel
    :param tokenizer: transformers tokenizer
    :param questions: 问题字符串列表
    :param max_new_tokens: 最大生成长度
    :param temperature: 采样温度
    :param top_p: nucleus sampling
    :return: 回答字符串列表，与 questions 顺序一致
    """
    # 1) 构造对话模板
    messages_list = [[{"role": "user", "content": q}] for q in questions]
    texts = [
        tokenizer.apply_chat_template(
            m,
            tokenize=False,
            add_generation_prompt=True
        )
        for m in messages_list
    ]

    # 2) tokenize & padding
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

    # 3) 生成
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )

    # 4) 解码
    answers = []
    for prompt_ids, gen_ids in zip(inputs.input_ids, gen):
        new_ids = gen_ids[len(prompt_ids):]
        answers.append(tokenizer.decode(new_ids, skip_special_tokens=True).strip())
    return answers


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Qwen2.5-3B-Instruct + LoRA 批量推理"
    )
    parser.add_argument(
        "-q", "--questions",
        nargs="+",
        default=[
            "快手电商搜索GMV同比提升了多少百分比",
            "京东集团第二季度物流分部的经营利润率创造了什么记录",
            "介绍一下量子计算。",
            "写一段 Python 快速排序代码。",
            "把“Hello world”翻译成日语。",
            "如何评价《三体》这部小说？",
            "美团的即时配送订单数达到了多少",
            "你是谁"
        ],
        help="要推理的问题列表"
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args(argv)

    # 简单校验路径
    if not Path(BASE_MODEL_PATH).exists():
        sys.exit(f"[ERROR] 基座模型路径不存在：{BASE_MODEL_PATH}")
    if not Path(LORA_ADAPTER_PATH).exists():
        sys.exit(f"[ERROR] LoRA 路径不存在：{LORA_ADAPTER_PATH}")

    model, tokenizer = load_model_and_tokenizer()

    print("\n========== 开始推理 ==========")
    answers = chat_generate(
        model,
        tokenizer,
        args.questions,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )

    for q, a in zip(args.questions, answers):
        print(f"\n【问题】{q}\n【回答】{a}\n{'-' * 50}")


if __name__ == "__main__":
    main()

    """
【问题】快手电商搜索GMV同比提升了多少百分比
【回答】快手电商搜索GMV同比提升了47%，这一增幅主要得益于直播电商业务的增长。在直播电商方面，快手电商通过优化用户体验和提升商品质量，推动了GMV的增长，从而实现了GMV同比增长25%。
--------------------------------------------------

【问题】京东集团第二季度物流分部的经营利润率创造了什么记录
【回答】京东集团第二季度物流分部的经营利润率创造了历史新高，达到了23.6%。这一成绩得益于高效的运营效率和成本控制，使得物流分部不仅能够维持稳定的收入增长，还能在利润方面取得显著提升。
--------------------------------------------------

【问题】介绍一下量子计算。
【回答】量子计算是一种基于量子力学原理的计算模型，它利用量子位（qubits）来存储和处理信息。与传统计算机使用的二进制位不同，量子位可以同时处于0和1的状态，这种特性称为叠加态。此外，量子位之间还可以通过量子纠缠实现更强的信息传递方式。这些特性使得量子计算机能够在解决某些特定问题时比经典计算机更为高效。

量子计算的研究领域包括量子算法开发、量子硬件设计以及量子纠错技术等。量子计算的目标是为那些无法用经典计算机有效求解的问题提供解决方案，例如大规模因式分解、优化问题和模拟量子系统等。尽管目前量子计算仍处在初级阶段，并且面临诸多挑战如噪声控制和可扩展性等问题，但其潜在的巨大潜力使其成为当今科技界研究的热点之一。
--------------------------------------------------

【问题】写一段 Python 快速排序代码。
【回答】```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

# 测试代码
if __name__ == "__main__":
    print(quick_sort([3,6,8,10,1,2,1]))
```

这段代码实现了快速排序算法，通过选择一个基准元素（pivot），将其余元素分成小于、等于和大于基准的三部分，并对这些部分进行递归地快速排序。最后将它们合并起来得到最终有序的数组。
--------------------------------------------------

【问题】把“Hello world”翻译成日语。
【回答】“Hello world”翻译成日语是“こんにちは、世界”。
--------------------------------------------------

【问题】如何评价《三体》这部小说？
【回答】《三体》是一部由刘慈欣所著的科幻小说，首次出版于2008年。该书荣获了第73届雨果奖最佳长篇故事奖，并被翻译成多国语言，受到全球读者的喜爱。它以独特的视角和丰富的想象力展现了人类与外星文明之间的冲突与合作，引发了人们对宇宙、生命以及未来科技的深刻思考。
--------------------------------------------------

【问题】美团的即时配送订单数达到了多少
【回答】美团的即时配送订单数达到了5亿单。
--------------------------------------------------

【问题】你是谁
【回答】我是杨天霸，一个由白沙科技公司开发的人工智能助手。我旨在通过对话方式回答用户的问题、提供解答以及进行各种互动。
--------------------------------------------------

    """
