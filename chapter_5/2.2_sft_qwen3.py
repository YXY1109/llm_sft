import logging
import os
import sys

import swanlab
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from swanlab.integration.transformers import SwanLabCallback
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel, is_bfloat16_supported

load_dotenv()

logger = logging.getLogger(__name__)
log_level = logging.INFO

swanlab_callback = SwanLabCallback(project="QWen3-8B")

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,  # 设置日志级别
    handlers=[
        logging.StreamHandler(sys.stdout),  # 输出到屏幕
        # logging.FileHandler("train.log")  # 输出到文件
    ]
)

logger.info("===========Login SwanLab====================")
api_key = os.getenv("SWANLAB_KEY")
swanlab.login(api_key="YOUR_SWANLAB_KEY")

max_seq_length = 8192
dtype = torch.bfloat16
load_in_4bit = False

logger.info("===========Loading the model and tokenizer=====================")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./models/Qwen/Qwen3-8B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    gpu_memory_utilization=0.95
)

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


def formatting_prompts_func(examples):
    inputs = examples["Question"]
    cots = examples["Complex_CoT"]
    outputs = examples["Response"]
    texts = []
    for input, cot, output in zip(inputs, cots, outputs):
        text = train_prompt_style.format(input, cot, output) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }


if __name__ == '__main__':
    logger.info("Test Code Running···")

    logger.info("==============Init train prompt====================")

    train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
    Write a response that appropriately completes the request. 
    Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

    ### Instruction:
    You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
    Please answer the following medical question. 

    ### Question:
    {}

    ### Response:
    <think>
    {}
    </think>
    {}"""

    logger.info("==============Load datasets====================")
    dataset = load_dataset("json", data_files="./medical_o1_sft_Chinese.json", split="train", trust_remote_code=True)
    dataset = dataset.map(formatting_prompts_func, batched=True, )

    train_test_split = dataset.train_test_split(test_size=0.1)

    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']

    logger.info(
        f"Dataset splited. Number of training samples = {len(train_dataset)}, number of test samples = {len(test_dataset)}")

    logger.info("==============Setting up the model====================")

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    sft_config = SFTConfig(
        dataset_text_field="text",  # 在这里指定文本字段
        max_seq_length=max_seq_length,
        dataset_num_proc=4,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        sft_config=sft_config,  # 传入SFT配置
        args=TrainingArguments(
            per_device_train_batch_size=16,
            gradient_accumulation_steps=2,
            per_device_eval_batch_size=4,
            # Use num_train_epochs = 1, warmup_ratio for full training runs!
            warmup_steps=50,
            max_steps=600,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir="qwen3_outputs",
            save_strategy="steps",
            save_steps=200,
            eval_strategy="steps",
            eval_steps=200
        ),
        callbacks=[swanlab_callback],
    )

logger.info("==============Model training====================")

trainer_stats = trainer.train()
try:
    logger.info("==============Save  model====================")
    MODEL_NAME = 'QWen3-8B-Medical-CoT'
    # model.save_pretrained(MODEL_NAME) 
    logger.info("==============2====================")
    # tokenizer.save_pretrained(MODEL_NAME)
    logger.info("==============3====================")
    model.save_pretrained_merged(MODEL_NAME, tokenizer, save_method="merged_16bit")
    logger.info("==============4====================")
except Exception as e:
    logger.info(e)
