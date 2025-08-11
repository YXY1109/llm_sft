from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 本地模型路径，替换为你实际的模型存放路径
model_path = r"D:\PycharmProjects\LLaMA-Factory\saves\Qwen3-0.6B-Instruct\Qwen3-0.6B-Instruct-yxy"


def run_modelscope_yxy():
    # 创建文本生成管道
    text_generation = pipeline(
        task=Tasks.text_generation,
        model=model_path,
        device="cuda"  # 使用 GPU 推理，如无 GPU 可改为 "cpu"
    )

    # 输入提示文本
    # prompt = "请介绍一下人工智能的发展历程"
    prompt = "你是谁"

    # 进行推理
    result = text_generation(
        prompt,
        max_length=512,  # 最大生成长度
        temperature=0.7,  # 温度参数，控制生成的随机性
        top_p=0.85,  # Top-p采样参数
        do_sample=True  # 是否使用采样策略
    )

    # 输出结果
    print("模型生成结果：")
    print(result)


def run_modelscope():
    model_name = model_path

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    # prepare the model input
    prompt = "你是谁"
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True  # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content)
    print("content:", content)


if __name__ == '__main__':
    run_modelscope_yxy()
    run_modelscope()
