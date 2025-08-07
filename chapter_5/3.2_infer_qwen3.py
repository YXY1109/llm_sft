import logging
import sys

from unsloth import FastLanguageModel

logger = logging.getLogger(__name__)
log_level = logging.INFO

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,  # 设置日志级别
    handlers=[
        logging.StreamHandler(sys.stdout),  # 输出到屏幕
        # logging.FileHandler("app.log")  # 输出到文件
    ]
)

max_seq_length = 4096
dtype = None
load_in_4bit = False

question = "患者为40岁男性，头疼、发烧，还有腹泻，持续2天了。"

prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
    Write a response that appropriately completes the request. 
    Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

    ### Instruction:
    You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
    Please answer the following medical question. 

    ### Question:
    {}

    ### Response:
    <think>{}"""

logger.info("======================Input question=====================")
logger.info(prompt_style.format(question, ""))

logger.info("===========Loading the model and tokenizer=====================")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./QWen3-8B-Medical-CoT",
    # model_name = "/model/Qwen/Qwen3-8B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    gpu_memory_utilization=0.95
)

logger.info("===========Infering=====================")

FastLanguageModel.for_inference(model)  # Unsloth has 2x faster inference!
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=4096,
    use_cache=True,
)
response = tokenizer.batch_decode(outputs)

logger.info(response[0].split("### Response:")[1])
