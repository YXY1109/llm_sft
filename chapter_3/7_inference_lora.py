import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


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


root_path = os.path.dirname(os.path.abspath(__file__))
model_name = os.path.join(root_path, "models/Qwen/Qwen3-1.7B")

# 加载原下载路径的tokenizer和model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)

# 加载lora模型
lora_name = os.path.join(root_path, "models/train_lora/Qwen3-1.7B/checkpoint-1084")
model = PeftModel.from_pretrained(model, model_id=lora_name)

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
<think>嗯，用户问的是糖尿病患者应该选择哪些碳水化合物，还提到自己被诊断为糖尿病，想知道该选哪些。首先，我需要回忆一下糖尿病管理中的碳水化合物选择原则。记得糖尿病管理的关键是控制血糖，所以选择低升糖指数（GI）的碳水化合物很重要。

首先，低GI食物通常指的是那些在血糖上升速度较慢的食物，比如全谷物、杂豆类、绿叶蔬菜这些。比如燕麦、糙米、藜麦这些全谷物，它们含有更多的膳食纤维，帮助减缓碳水化合物的消化和吸收，从而减缓血糖上升速度。然后，杂豆类比如扁豆、鹰嘴豆，虽然可能GI稍高一些，但整体上还是相对低GI的，而且富含蛋白质和纤维，有助于控制血糖。

接下来，绿叶蔬菜和水果也是不错的选择。比如菠菜、羽衣甘蓝这些蔬菜，它们的GI非常低，而且富含维生素和矿物质，有助于整体健康。水果的话，像苹果、梨这些，虽然GI略高，但选择低糖分的品种，比如草莓、蓝莓，可能更适合。不过要注意控制摄入量，避免血糖骤升。

然后，用户可能还关心如何具体选择这些食物。比如全谷物面包、糙米、燕麦这些主食，建议每天摄入多少克？可能需要给出具体的建议，比如每天50-75克全谷物，或者每天2-3种蔬菜。另外，杂豆类每天100克左右，绿叶蔬菜每天200-300克，水果每天100-200克，这样分量合适。

还要考虑烹饪方式，比如煮熟的燕麦、蒸煮的蔬菜，这些方式更健康，避免高油高盐。另外，是否要避免高GI的碳水化合物，比如白面包、白米饭、甜点等，这些应该排除在饮食之外。

用户可能还担心饮食中的其他因素，比如脂肪和蛋白质的摄入。需要强调低GI碳水化合物的同时，也要注意蛋白质的摄入，比如瘦肉、鸡蛋、豆制品，这些有助于维持饱腹感，减少饥饿感，从而减少对高GI碳水的依赖。

另外，可能需要提醒用户遵循医生或营养师的个性化建议，因为每个人的具体情况不同，比如病情阶段、血糖目标、并发症等，可能需要调整碳水摄入量和类型。所以建议用户定期监测血糖，并与医疗团队沟通调整饮食方案。

有没有可能用户没有考虑到其他因素？比如，是否要限制某些碳水来源，或者是否需要选择特定的食品品牌？不过问题主要集中在碳水化合物的选择，所以可能不需要深入这些细节。但可以建议用户选择包装健康、无添加的食品，避免加工过的高GI产品。

总结下来，回答的结构应该是先推荐低GI食物，分点说明哪些食物，然后给出具体的分量建议，最后强调遵循医生指导的重要性。这样用户能清楚了解应该选择哪些食物，并知道如何合理分配摄入量。
</think> 
 您好，对于糖尿病患者来说，选择低升糖指数的碳水化合物非常重要。这类碳水化合物包括全谷物、杂豆类、绿叶蔬菜以及水果。这些食物有助于控制血糖水平，同时提供必要的营养。例如，您可以选择每天50到75克的全谷物，每天2到3种杂豆类，每天200到300克的绿叶蔬菜，以及每天100到200克的水果。此外，还可以考虑选择一些低GI的主食，如燕麦、糙米等，来替代高GI的主食。建议您咨询专业的营养师或医生，根据您的具体情况调整饮食计划，确保饮食既健康又符合您的需求。
"""
