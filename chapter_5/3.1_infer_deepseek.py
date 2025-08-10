import logging
import sys
from typing import Dict, List, Tuple

import torch
from unsloth import FastLanguageModel

# 配置日志
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("medical_inference.log")  # 同时输出到文件
    ]
)

# 模型配置
MAX_SEQ_LENGTH = 4096
DTYPE = torch.bfloat16
LOAD_IN_4BIT = True

# 模型路径
BASE_MODEL_PATH = "./models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
FINETUNED_MODEL_PATH = "./Deepseek-R1-Medical-CoT"

# 提示词模板
PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
Please answer the following medical question. 

### Question:
{}

### Response:
"""

# 测试问题集合 - 包含不同类型的医学问题
TEST_QUESTIONS = [
    "患者为40岁男性，头疼、发烧，还有腹泻，持续2天了。可能的诊断是什么？应该如何处理？",
    "35岁女性患者，主诉持续咳嗽三周，伴有胸闷和轻微发热。无吸烟史，近期未旅行。可能的病因是什么？",
    "60岁男性，有高血压病史，突然出现左侧肢体无力和言语不清，症状已持续30分钟。最可能的诊断是什么？应立即采取什么措施？",
    "25岁女性，出现尿频、尿急、尿痛症状2天，无发热。可能的诊断是什么？推荐的治疗方案是什么？",
    "50岁男性，肥胖，近几个月出现口渴、多尿、体重下降。可能的诊断是什么？需要做哪些检查来确认？",
    "65岁女性，糖尿病史，突发胸痛伴大汗30分钟。",
    "3岁儿童误服漂白剂后出现呛咳、呕吐，如何处理？"
]


def load_model(model_path: str) -> Tuple[torch.nn.Module, object]:
    """加载指定路径的模型和分词器"""
    try:
        logger.info(f"正在加载模型: {model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
            gpu_memory_utilization=0.95
        )
        FastLanguageModel.for_inference(model)  # 启用推理模式
        logger.info(f"模型 {model_path} 加载成功")
        return model, tokenizer
    except Exception as e:
        logger.error(f"加载模型 {model_path} 失败: {str(e)}", exc_info=True)
        raise


def generate_response(model: torch.nn.Module, tokenizer: object, question: str, max_new_tokens: int = 4096) -> str:
    """使用模型生成回答"""
    try:
        prompt = PROMPT_TEMPLATE.format(question)
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        # 提取响应部分，去掉提示词
        if "### Response:" in response:
            response = response.split("### Response:")[1].strip()
        return response
    except Exception as e:
        logger.error(f"生成回答时出错: {str(e)}", exc_info=True)
        return f"生成回答失败: {str(e)}"


def run_comparison(base_model: Tuple[torch.nn.Module, object],
                   finetuned_model: Tuple[torch.nn.Module, object],
                   questions: List[str]) -> List[Dict]:
    """运行模型对比测试"""
    results = []
    base_model_obj, base_tokenizer = base_model
    finetuned_model_obj, finetuned_tokenizer = finetuned_model

    for i, question in enumerate(questions, 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"问题 {i}/{len(questions)}: {question}")
        logger.info(f"{'-' * 80}")

        # 基础模型推理
        logger.info("基础模型生成回答中...")
        base_response = generate_response(base_model_obj, base_tokenizer, question)

        # 微调模型推理
        logger.info("微调模型生成回答中...")
        finetuned_response = generate_response(finetuned_model_obj, finetuned_tokenizer, question)

        # 存储结果
        results.append({
            "question": question,
            "base_model_response": base_response,
            "finetuned_model_response": finetuned_response
        })

        # 显示结果，便于对比
        logger.info(f"\n{'-' * 20} 基础模型回答 {'-' * 20}")
        logger.info(base_response)
        logger.info(f"\n{'-' * 20} 微调模型回答 {'-' * 20}")
        logger.info(finetuned_response)
        logger.info(f"\n{'=' * 80}")

    return results


def save_results(results: List[Dict], filename: str = "model_comparison_results.txt"):
    """保存对比结果到文件，方便后续分析"""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            for i, result in enumerate(results, 1):
                f.write(f"{'=' * 100}\n")
                f.write(f"问题 {i}: {result['question']}\n")
                f.write(f"{'-' * 100}\n")
                f.write(f"基础模型回答:\n{result['base_model_response']}\n\n")
                f.write(f"{'-' * 100}\n")
                f.write(f"微调模型回答:\n{result['finetuned_model_response']}\n")
                f.write(f"{'=' * 100}\n\n")
        logger.info(f"结果已保存到 {filename}")
    except Exception as e:
        logger.error(f"保存结果失败: {str(e)}", exc_info=True)


def main():
    try:
        # 加载模型
        base_model = load_model(BASE_MODEL_PATH)
        finetuned_model = load_model(FINETUNED_MODEL_PATH)

        # 运行对比测试
        results = run_comparison(base_model, finetuned_model, TEST_QUESTIONS)

        # 保存结果到文件
        save_results(results)

    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
