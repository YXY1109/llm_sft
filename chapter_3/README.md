# Qwen3微调实战：医疗R1推理风格聊天

- **基础模型**：[Qwen3-1.7B](https://modelscope.cn/models/Qwen/Qwen3-1.7B/summary)
- **数据集**：[delicate_medical_r1_data](https://modelscope.cn/datasets/krisfu/delicate_medical_r1_data)
- **SwanLab
  **：[qwen3-sft-medical-all](https://swanlab.cn/@YXY1109/qwen3-sft-medical_all/runs/tfoeswcdth20v55yb0mtq/chart)
- **SwanLab**：[qwen3-sft-medical-loar]()
- **微调方式**：全参数微调、LoRA微调
- **推理风格**：R1推理风格

> 如果需要进一步降低显存需求，可以使用Qwen3-0.6B模型，或调低`MAX_LENGTH`。

## 安装环境

```bash
pip install -r requirements.txt
```

## 数据准备

会自动完成数据集下载、预处理、验证集划分，生成`train.jsonl`和`val.jsonl`文件。

```bash
python 1_generate_data.py
```

## 下载模型

```bash
python 2_download_model.py
```

## 原始模型推理

```bash
python 3_inference.py
```

## 训练

**全参数微调**

```bash
python 4_train_all.py
```

**LoRA微调**

```bash
python 5_train_lora.py
```

## 训练效果预览

```
Question: 
医生，我最近胃部不适，听说有几种抗溃疡药物可以治疗，您能详细介绍一下这些药物的分类、作用机制以及它们是如何影响胃黏膜的保护与损伤平衡的吗？

LLM:
<think>
嗯，用户问的是抗溃疡药物的分类、作用机制，以及它们如何影响胃黏膜的保护和损伤平衡。
首先，我需要回忆一下抗溃疡药物的主要分类。记得以前学过，抗溃疡药物通常分为几大类，比如抑酸药、胃黏膜保护剂、促胃动力药和抗幽门螺杆菌药物。可能还有其他类型，比如吸附剂之类的，但可能用户的问题里没有提到，所以可能不需要展开。
接下来，每个类别的具体药物例子。比如抑酸药，常见的有H2受体拮抗剂和质子泵抑制剂。比如雷尼替丁、奥美拉唑这些。作用机制是抑制胃酸分泌，特别是H2受体拮抗剂通过阻断组胺引起的胃酸分泌，而质子泵抑制剂直接作用于胃酸分泌的最后一步，抑制质子泵。这部分需要详细说明。
胃黏膜保护剂的话，比如硫糖铝、铋剂，它们的作用是形成保护层，或者促进黏液分泌，比如硫糖铝可能通过黏附在黏膜表面形成保护膜，而铋剂可能促进黏液分泌，同时可能有中和胃酸的作用？或者可能主要是中和作用？需要确认。另外，胶体果胶铋可能也是例子。
促胃动力药比如多潘立酮、西沙必利，作用是增强胃蠕动，减少胃酸反流，这样胃排空快，可能减少溃疡形成。但用户的问题里提到的是促进胃排空，所以这部分需要说明。
抗幽门螺杆菌的药物通常包括抗生素，比如阿莫西林、克拉霉素，但抗幽门螺杆菌药物可能还有三联或四联疗法，比如加上PPI和铋剂。需要提到这些药物的作用机制是抑制幽门螺杆菌的生长，比如抗生素杀灭细菌，而PPI可能同时抑制胃酸分泌，但如果是抗幽门螺杆菌药物的话，可能是指专门针对该病的药物，比如可能还有铋剂或者其他药物？
不过用户的问题里提到抗幽门螺杆菌药物，可能需要明确是单独针对幽门螺杆菌，还是包括抗生素和PPI。可能需要指出，抗幽门螺杆菌药物通常包括抗生素和PPI，而PPI本身是抑酸药。所以可能需要说明这些药物如何通过抑制胃酸分泌和杀灭细菌来减少溃疡。
然后，关于作用机制如何影响胃黏膜的保护和损伤平衡。比如，抑酸药减少胃酸，从而减少对黏膜的侵蚀，同时保护剂形成保护层，促进黏液分泌，增强黏膜屏障。促胃动力药通过增加胃蠕动，减少胃酸反流，从而保护黏膜。而抗幽门螺杆菌药物通过减少胃酸分泌和杀灭细菌，减少损伤因素。
可能需要检查是否有遗漏的药物分类，比如吸附剂如活性炭，或者中和剂如氢氧化铝，但用户的问题可能只关注主要的四类。另外，是否需要提到铋剂的其他作用，比如促进黏液分泌和前列腺素合成？比如胶体果胶铋可能含有前列腺素，增强黏液分泌。
在解释机制时，要确保每个药物的作用点和如何影响保护和损伤。例如，抑酸药直接减少胃酸，而保护剂形成物理屏障。促胃动力药通过改变胃排空模式，减少反流，从而减少损伤。抗幽门螺杆菌药物通过减少胃酸和杀灭病原体，两者都减少损伤。
可能需要确认每个药物的正确分类和例子是否正确，比如西沙必利是否属于促胃动力药，或者是否属于H2受体拮抗剂？西沙必利其实是属于促胃动力药，因为它的作用是增强胃排空。而奥美拉唑是PPI，属于抑酸药。需要确保这些分类正确。
另外，用户提到自己胃部不适，可能需要建议就医，但答案里已经包含这点，所以没问题。
总结下来，结构应该是先分类，再每个类别下的药物例子和作用机制，然后说明这些机制如何影响保护和损伤平衡。需要确保每个部分都清晰，逻辑连贯，并且覆盖用户的问题点。
</think> 

当然可以。抗溃疡药物主要分为四类：抑酸药、胃黏膜保护剂、促胃动力药和抗幽门螺杆菌药物。抑酸药如H₂受体拮抗剂和质子泵抑制剂，它们的作用是减少胃酸的分泌，从而减轻对胃黏膜的损伤。胃黏膜保护剂如硫糖铝和胶体果胶铋，它们能帮助形成一层保护膜，覆盖在胃黏膜表面，减少胃酸和胃蛋白酶对黏膜的侵蚀。促胃动力药如多潘立酮和西沙必利，通过增强胃部的蠕动，减少胃内食物的滞留时间，从而降低溃疡的风险。抗幽门螺杆菌药物则通过抑制幽门螺杆菌的生长，减少其对胃黏膜的伤害。这些药物的共同目标是通过不同的机制来平衡胃黏膜的保护因素和损伤因素，帮助溃疡愈合。如果您有具体的症状或担心某种药物，可以告诉我，我会为您推荐合适的治疗方案。
```

## 推理

**全参数微调**

- 这个和3_inference.py一样，只是模型位置参数不同

```bash
python 6_inference_all.py

```

**LoRA微调**

```bash
python 6_inference_lora.py
```

