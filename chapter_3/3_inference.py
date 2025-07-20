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
model_name = os.path.join(root_path, "models/Qwen/Qwen3-1.7B")

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
<think>
好的，用户最近被诊断为糖尿病，需要选择适合的碳水化合物。首先，我得回忆一下糖尿病患者的饮食管理原则。糖尿病患者需要控制碳水化合物的摄入，但具体选择哪种类型很重要。

用户可能不了解不同碳水化合物的升糖指数（GI）差异。比如，精制碳水化合物如白米饭、白面包GI高，容易导致血糖迅速上升，而全谷物、豆类、蔬菜等GI低，更合适。需要解释这些差异，并给出具体例子。

另外，用户可能关心如何分配碳水化合物，比如控制总摄入量，分次进食，避免血糖剧烈波动。可能还需要提到膳食纤维丰富的食物，如燕麦、豆类，这些能延缓碳水化合物吸收，帮助稳定血糖。

还要考虑用户是否有其他健康问题，比如高血压或高血脂，是否需要额外注意某些碳水化合物的摄入。不过用户目前只提到糖尿病，所以重点放在碳水化合物选择上。

可能需要建议用户咨询营养师制定个性化饮食计划，因为每个人的情况不同。同时，提醒用户避免过量摄入精制碳水，保持均衡饮食，包括蛋白质、脂肪和蔬菜。

最后，确保回答清晰、有条理，用易懂的语言，避免专业术语过多，让用户容易理解和执行。
</think>

对于糖尿病患者来说，碳水化合物的摄入需要科学规划，以维持血糖稳定。以下是关键建议：

---

### **1. 选择低升糖指数（GI）的碳水化合物**
- **低GI碳水化合物**：  
  - **全谷物**：如糙米、燕麦、全麦面包、藜麦（升糖缓慢，血糖波动小）。  
  - **豆类**：如黑豆、鹰嘴豆、扁豆（富含膳食纤维，延缓碳水化合物吸收）。  
  - **蔬菜**：如菠菜、胡萝卜、西兰花（含膳食纤维，升糖缓慢）。  
  - **水果**：如苹果、梨、莓类（含纤维和抗氧化物质，但需控制量）。  

- **高GI碳水化合物**：  
  - **精制碳水**：如白米、白面包、甜点、蛋糕（需严格控制摄入量）。  
  - **加工食品**：如薯片、糖果、含糖饮料（升糖快，需避免）。

---

### **2. 控制碳水化合物的摄入总量和分配**
- **总热量管理**：根据个人基础代谢率和活动量，制定每日碳水化合物的总摄入量（通常占总热量的45%-60%）。  
- **分次摄入**：建议将碳水化合物分成2-3次餐次，避免血糖剧烈波动（如早餐吃主食，午餐搭配蛋白质和蔬菜，晚餐少量碳水）。  
- **避免“空腹碳水”**：避免在空腹时大量摄入碳水，可将碳水分配在三餐中。

---

### **3. 增加膳食纤维和蛋白质**
- **膳食纤维**：帮助延缓碳水化合物吸收，稳定血糖。  
  - **来源**：全谷物、豆类、蔬菜、坚果。  
- **优质蛋白**：如鱼、鸡胸肉、豆腐、鸡蛋，可延缓碳水化合物的消化吸收，减少血糖波动。  
- **健康脂肪**：如坚果、橄榄油，有助于控制胰岛素分泌。

---

### **4. 避免“隐性碳水”**
- **警惕隐藏的碳水**：  
  - 食用油（如花生油、橄榄油）含少量碳水（约1%）。  
  - 酸奶、果汁含糖分（如含糖酸奶、果汁）。  
  - 调味品（如糖浆、酱料）可能含高糖。

---

### **5. 个性化调整**
- **根据血糖监测结果调整**：  
  - 若血糖波动大，可增加低GI碳水比例（如将主食替换为糙米）。  
  - 若血糖稳定，可适当减少碳水摄入量（如控制在150-200克/天）。  
- **咨询营养师**：制定个性化饮食方案，避免盲目节食或过度摄入。

---

### **示例饮食搭配（每日）**
- **早餐**：燕麦粥（50g燕麦）+ 1个水煮蛋 + 1个苹果  
- **午餐**：糙米饭（100g）+ 鸡胸肉（100g）+ 西兰花（100g）  
- **晚餐**：藜麦（50g）+ 豆腐（100g）+ 番茄炒蛋  
- **加餐**：一小把坚果（15g）或低糖水果（如蓝莓）

---

### **注意事项**
- **避免“碳水炸弹”**：如蛋糕、饼干、甜点等高糖高GI食物。  
- **多喝水**：每日1500-2000ml，避免含糖饮料。  
- **定期监测血糖**：根据数据调整饮食，避免“一刀切”。

---

通过合理选择低GI碳水化合物、控制总量和分配，结合均衡饮食和规律运动，可以有效管理糖尿病，减少并发症风险。建议在医生或营养师指导下制定个性化方案。
"""
