from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

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
[INFO] 自动选择设备: cuda
[INFO] 加载 tokenizer...
[INFO] 加载基座模型...
Loading checkpoint shards: 100%|██████████| 2/2 [00:14<00:00,  7.23s/it]
[INFO] 加载 LoRA adapter...

========== 开始推理 ==========

【问题】快手电商搜索GMV同比提升了多少百分比
【回答】快手电商搜索GMV同比提升了26%。这一增长主要得益于快手小店、快品牌及快手直播三大业务板块的共同推动，其中，快手小店的GMV同比增长了14%，快品牌的GMV同比增长了47%，而快手直播的GMV同比增长了51%。
--------------------------------------------------

【问题】京东集团第二季度物流分部的经营利润率创造了什么记录
【回答】京东集团第二季度物流分部的经营利润率创造了2015年以来的新高。
--------------------------------------------------

【问题】介绍一下量子计算。
【回答】量子计算是一种基于量子力学原理的新型计算技术，它利用量子比特（qubits）来存储和处理信息，这些量子比特可以同时表示多种状态，从而实现并行计算的能力。与经典计算机使用二进制位（bits）进行计算不同，量子计算机利用量子比特可以同时表示0和1的状态，这种特性使得量子计算机在处理特定问题时比传统计算机更高效。

量子计算的关键优势包括：

1. 并行性：量子计算机可以在多个量子比特上同时执行计算任务，从而极大地提高了计算效率。
2. 算法效率：某些算法在量子计算机上的运行速度远远超过经典计算机，例如Shor算法可以快速分解大整数，而经典算法需要的时间则呈指数增长。
3. 处理大规模数据的能力：量子计算机能够高效地处理大规模数据集，这对于解决复杂的优化问题、模拟分子结构等方面具有重要意义。

尽管量子计算目前仍处于发展阶段，但其潜力巨大。随着技术的进步和更多应用场景的研究，未来量子计算机有望为科学研究、材料设计、药物发现等领域带来革命性的变化。
--------------------------------------------------

【问题】写一段 Python 快速排序代码。
【回答】当然可以，以下是一个使用 Python 实现的快速排序算法：

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

# 测试快速排序函数
test_array = [3,6,8,10,1,2,1]
print("原始数组:", test_array)
sorted_array = quick_sort(test_array)
print("排序后的数组:", sorted_array)
```

这段代码定义了一个名为 `quick_sort` 的函数，它接受一个列表作为参数并返回排序后的列表。这个实现采用了递归的方法来对列表进行排序，并且通过选择一个基准元素（在这个例子中是列表中间的元素）来将列表分成三个部分：小于基准、等于基准和大于基准的元素。然后，递归地对左右两边的部分进行排序，最后将这三个部分合并起来得到最终的排序结果。
--------------------------------------------------

【问题】把“Hello world”翻译成日语。
【回答】"Hello world" 翻译成日语是 "こんにちは、世界" (Konnichiwa, sekai)。
--------------------------------------------------

【问题】如何评价《三体》这部小说？
【回答】《三体》是一部由刘慈欣所著的中国科幻小说，首次出版于2008年。这部作品因其深刻的哲学思考、复杂的情节结构以及对人类文明的深刻洞察而受到了广泛的关注和赞誉。它不仅在国内赢得了极高的人气，还获得了雨果奖这一世界级科幻文学奖项的认可，成为了中国科幻文学的里程碑之作。《三体》通过讲述地球文明与三体文明之间的互动故事，探讨了宇宙间的文明冲突与合作、科技发展与道德伦理等问题，展现了宏大的想象力和深厚的文化底蕴。
--------------------------------------------------
    """
