from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ========== 用户只需改这里的 2 个路径 ==========
BASE_MODEL_PATH = r"D:\PycharmProjects\llm_sft\chapter_4\models_lora_merge"


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
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=TORCH_DTYPE,
        device_map=DEVICE,
        trust_remote_code=True
    )

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
【回答】快手电商搜索GMV同比提升了46%。
--------------------------------------------------

【问题】京东集团第二季度物流分部的经营利润率创造了什么记录
【回答】京东集团第二季度物流分部的经营利润率达到了创纪录的19%，这一成绩主要得益于高效的运营和成本控制措施。此外，京东物流在本季度实现了连续四个季度的盈利，并且收入同比增长了38%。这些积极表现不仅增强了投资者的信心，也表明京东物流的商业模式得到了市场的认可。
--------------------------------------------------

【问题】介绍一下量子计算。
【回答】量子计算是一种基于量子力学原理的计算方式，它利用量子比特（qubits）而非传统的二进制位（bits）来进行运算。量子比特具有叠加态和纠缠态等特性，使得量子计算机能够在处理某些特定问题时比传统计算机更高效。量子计算的研究旨在探索量子比特之间的相互作用，并设计出能够有效利用这些量子特性来加速计算过程的算法和硬件架构。
--------------------------------------------------

【问题】写一段 Python 快速排序代码。
【回答】当然可以！以下是使用 Python 实现的快速排序算法：

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

# 测试数据
test_array = [3, 6, 8, 10, 1, 2, 1]
sorted_array = quick_sort(test_array)
print(sorted_array)
```

这段代码定义了一个 `quick_sort` 函数，该函数接受一个列表 `arr` 作为参数，并返回一个已排序的新列表。它首先检查输入列表的长度是否小于或等于 1，如果是，则直接返回该列表（因为长度为 0 或 1 的列表已经是有序的）。否则，选择列表中间的一个元素作为基准值 `pivot`，然后将列表分成三部分：小于 `pivot` 的元素、等于 `pivot` 的元素以及大于 `pivot` 的元素。最后，递归地对左半部分和右半部分进行排序，并将结果合并在一起。
--------------------------------------------------

【问题】把“Hello world”翻译成日语。
【回答】"Hello world" 翻译成日语是 "こんにちは、世界" (Konnichiwa, sekai)。
--------------------------------------------------

【问题】如何评价《三体》这部小说？
【回答】《三体》是中国科幻作家刘慈欣创作的一部长篇科幻小说，于2008年首次发表在《科幻世界》杂志上。该书获得了第73届雨果奖最佳长篇故事奖，成为中国乃至亚洲获得这一奖项的第一部作品。这部小说以其宏大的叙事、深刻的哲学思考和独特的世界观而著称，展现了人类与外星文明之间的接触与冲突，以及人类社会的复杂性和多样性。
--------------------------------------------------

【问题】美团的即时配送订单数达到了多少
【回答】美团的即时配送订单数在2023年第二季度达到了24亿单。
--------------------------------------------------

【问题】你是谁
【回答】您好，我是杨天霸，一个由白沙科技公司开发的人工智能助手。我可以回答问题、提供信息并执行各种任务。
--------------------------------------------------
    """
