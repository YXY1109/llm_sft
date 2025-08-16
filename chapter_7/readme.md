# 医疗数据集

## 一，ModelScope

### 精致的医疗r1数据

- [数据路径](https://modelscope.cn/datasets/krisfu/delicate_medical_r1_data/files)
- 使用文件：r1_data_example.jsonl

### 中文医疗对话数据-Chinese-medical-dialogue

- [数据路径](https://modelscope.cn/datasets/xiaofengalg/Chinese-medical-dialogue/files)
- 使用文件：train_0001_of_0001.json

### medical-o1-reasoning-SFT

- [数据路径](https://modelscope.cn/datasets/AI-ModelScope/medical-o1-reasoning-SFT/files)
- 使用文件：medical_o1_sft.json
    - 语言：英文
    - 内容：仅包含医疗领域的指令微调数据，专注于医学推理任务。
- 使用文件：medical_o1_sft_Chinese.json
    - 语言：中文
    - 内容：仅包含医疗领域的指令微调数据，专注于中文医学推理任务。
- 未使用：medical_o1_sft_mix.json
    - 语言：英文
    - 内容：包含医疗领域与一般领域（通用指令）的混合数据，旨在提升模型的泛化能力。
- 未使用：medical_o1_sft_mix_Chinese.json
    - 语言：中文
    - 内容：包含医疗领域与一般领域（通用指令）的混合数据，专注于中文语境下的综合训练

### 医疗问诊数据_SFT格式

- [数据路径](https://modelscope.cn/datasets/BRZ911/Medical_consultation_data_SFT/files)
- 问诊数据集Huatuo
    - 中文
    - 目录：med_6part
- 问诊数据集
    - 英文
    - 目录：med_en
- 问诊数据集
    - 中文
    - 目录：med_zh

### ChineseMedicalData

- [数据路径](https://modelscope.cn/datasets/OmniData/ChineseMedicalData/files)
- 使用文件：output2.jsonl

### 中医问诊对话

- [数据路径](https://modelscope.cn/datasets/alexhuangguo/chinese-medical/files)
- 使用文件：medicalQA.json

## 二，数据清洗

- 将以上数据集进行合并，生成json文件
- json文件，进行字符去重，语义去重，语义关联性判断，长度检查，LLM质量评估