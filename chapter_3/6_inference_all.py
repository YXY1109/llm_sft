import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def predict(messages, model, tokenizer):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=2048)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


# 原始模型路径
root_path = os.path.dirname(os.path.abspath(__file__))
model_name = os.path.join(root_path, "models/train_all/Qwen3-1.7B/checkpoint-1084")

# 加载原下载路径的tokenizer和model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)

test_texts = {
    'instruction': "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。",
    'input': "医生，我最近被诊断为糖尿病，听说碳水化合物的选择很重要，我应该选择什么样的碳水化合物呢？"
}

instruction = test_texts['instruction']
input_value = test_texts['input']

messages = [
    {"role": "system", "content": f"{instruction}"},
    {"role": "user", "content": f"{input_value}"}
]

response = predict(messages, model, tokenizer)
print(response)
"""
<think>嗯，用户问的是糖尿病患者应该选择哪些类型的碳水化合物，还提到自己被诊断为糖尿病，想知道应该选什么样的。首先，我需要回忆一下糖尿病饮食的基本原则。记得糖尿病患者需要控制碳水化合物的摄入，但具体怎么选呢？

首先，应该考虑血糖生成指数（GI）的概念。高GI的食物会导致血糖快速升高，而低GI的食物则相反，能更缓慢地提升血糖。所以，答案里提到的低GI食物肯定是关键。比如全谷物、杂豆这些，可能比精制的碳水化合物更好。

然后，用户可能想知道具体应该避免哪些。答案里提到了精制糖和白面包，这些通常GI值高，容易导致血糖波动。所以需要强调避免这些，选择更复杂的碳水化合物。

另外，可能要考虑食物的其他因素，比如营养成分。全谷物和杂豆含有更多的纤维和B族维生素，这对糖尿病患者来说是有益的，因为纤维可以延缓碳水化合物的吸收，而B族维生素有助于代谢。而精制糖虽然GI高，但可能含有其他营养成分，不过糖尿病患者可能更关注血糖而非其他营养，但答案里还是建议避免。

用户可能还有潜在的需求，比如如何具体搭配这些食物，或者是否需要计算碳水化合物的量。但问题中没有明确问，所以可能不需要深入，但答案里已经涵盖了主要点，可能足够了。

还要注意用户可能混淆了不同碳水化合物的类型，比如是否所有全谷物都是低GI？比如白米和白面都是高GI，而全麦面包可能GI较低？需要确认这一点，但根据常见建议，全谷物和杂豆通常被认为是好的选择。

另外，是否需要提到复合碳水化合物和简单碳水化合物的区别？可能用户已经知道，但作为回答，可以简要说明复合碳水化合物的结构更复杂，消化吸收更慢。

总结下来，思考过程应该是先确定低GI的重要性，列举具体食物，然后指出避免哪些，最后提到其他好处。这样用户的问题就全面覆盖了，并且给出实用的建议。
</think> 
 您好，对于糖尿病患者来说，选择低血糖生成指数（GI）的食物是非常重要的。这类食物包括全谷物、杂豆、粗粮制品以及一些蔬菜，它们的血糖生成指数通常低于精制糖和白面包。选择这些食物可以更好地控制血糖水平，减少血糖波动。同时，这些食物中还含有丰富的营养成分，如膳食纤维、维生素和矿物质，对您的健康非常有益。希望这些建议对您有所帮助。
"""
