# LLM模型训练全流程实战教程

> 从零开始掌握大语言模型训练、微调、评估与部署的完整技术栈

## 📚 项目概述

本项目是一个系统性的LLM（大语言模型）训练学习仓库，涵盖了从基础理论到实际应用的完整技术链路。通过8个循序渐进的章节，深入探索不同规模、不同领域、不同技术栈的模型训练方法，为研究者提供全面的技术参考和实践指导。

### 🎯 学习目标

- 掌握从零开始训练大语言模型的完整流程
- 熟练运用多种微调框架（Transformers、LLaMA-Factory、Unsloth）
- 深入理解医疗、金融等垂直领域的模型应用
- 学会构建数据清洗、模型评估、部署发布的完整工具链

## 🗂️ 章节结构

### Chapter 1: 从零训练模型
**核心技术栈**: MiniMind + AutoDL
**学习重点**:
- 从零开始训练语言模型
- 预训练与微调的完整流程
- 云端训练环境的搭建与优化

**关键文件**:
- `chapter_1/json_demo.py`: JSON数据处理示例
- `chapter_1/minimind学习教程.md`: MiniMind框架详细教程

---

### Chapter 2: 数据集构建与LLaMA-Factory实战
**核心技术栈**: Easy-Data + LLaMA-Factory + HuggingFace
**学习重点**:
- 高质量数据集的构建方法
- LLaMA-Factory低代码微调框架使用
- 模型合并、GGUF转换与Ollama部署

**应用场景**: 财经数据问答系统
**技术亮点**: 端到端的模型训练到部署流水线

---

### Chapter 3: 手写实现Qwen3医疗微调
**核心技术栈**: Transformers + 手写训练循环
**学习重点**:
- 深入理解Transformers训练原理
- 手写实现全参数微调和LoRA微调
- 医疗R1推理风格的Chain-of-Thought训练

**模型**: Qwen3-1.7B
**数据集**: [delicate_medical_r1_data](https://modelscope.cn/datasets/krisfu/delicate_medical_r1_data)
**训练方式**: 全参数微调 + LoRA微调对比实验

**关键脚本**:
```bash
# 数据预处理与模型准备
python chapter_3/1_generate_data.py
python chapter_3/2_download_model.py

# 训练脚本
python chapter_3/4_train_all.py      # 全参数微调
python chapter_3/5_train_lora.py     # LoRA微调

# 推理验证
python chapter_3/6_inference_all.py  # 全参数模型推理
python chapter_3/7_inference_lora.py  # LoRA模型推理
```

**实验监控**: [SwanLab训练追踪](https://swanlab.cn/@YXY1109/qwen3-sft-medical_all)

---

### Chapter 4: LLaMA-Factory财报数据微调
**核心技术栈**: LLaMA-Factory + Qwen2.5-3B-Instruct
**学习重点**:
- 低代码框架的高效使用
- 财经垂直领域的模型适配
- 配置化训练参数调优

**数据集**: [FinancialData-SecondQuarter-2024](https://github.com/llm-factory/FinancialData-SecondQuarter-2024)
**模型**: Qwen2.5-3B-Instruct

**关键特性**:
- 可视化配置界面
- 支持多种微调策略
- 丰富的评估指标

---

### Chapter 5: Unsloth高效微调实践
**核心技术栈**: Unsloth + Qwen3-8B + DeepSeek-R1-Distill
**学习重点**:
- Unsloth内存优化技术
- 大规模模型的高效训练
- 医疗CoT推理的深度优化

**模型规模**: 7B-8B参数级别
**数据集**: [medical-o1-reasoning-SFT](https://modelscope.cn/datasets/AI-ModelScope/medical-o1-reasoning-SFT)
**技术优势**:
- 显存使用量显著降低
- 训练速度大幅提升
- 支持4-bit量化训练

**实验监控**: [Qwen3-8B](https://swanlab.cn/@YXY1109/chapter_5/runs/n17u1fsglugt1s0ba2lq8/chart) | [DeepSeek](https://swanlab.cn/@YXY1109/chapter_5/runs/p6zgm0ljxdevk91kbohob/chart)

---

### Chapter 6: 模型评估体系构建
**核心技术栈**: EvalScope + ROUGE/BLEU + LLM评估
**学习重点**:
- 多维度模型性能评估
- 自动化评估流水线
- 评估结果的可视化分析

**评估模型**: Qwen2.5-0.5B, Qwen3-8B
**评估维度**:
- 知识问答能力（GSM8K）
- 逻辑推理能力（ARC）
- 文本生成质量（ROUGE/BLEU）
- LLM辅助评估

---

### Chapter 7: 医疗数据集清洗与处理
**核心技术栈**: Sentence-Transformers + Milvus + 语义去重
**学习重点**:
- 大规模数据清洗技术
- 语义相似度去重算法
- 向量数据库的应用
- 数据质量评估体系

**技术亮点**:
- 基于BERT的语义去重
- Milvus向量检索
- 自动化数据清洗流水线
- 数据质量评分系统

---

### Chapter 8: Transformers深入学习
**学习重点**:
- Transformers架构原理
- 高级训练技巧
- 性能优化策略
- 前沿技术跟踪

## 🛠️ 环境配置

### 基础环境要求
- Python 3.8+
- CUDA 11.0+ （GPU训练）
- 16GB+ GPU显存（推荐）

### 依赖安装
```bash
# Chapter 3: Transformers基础训练
pip install -r chapter_3/requirements.txt

# Chapter 5: Unsloth高效训练
pip install -r chapter_5/requirements.txt

# Chapter 7: 数据清洗处理
pip install -r chapter_7/requirements.txt
```

## 🚀 快速开始

### 1. 选择学习路径
根据基础水平选择合适的学习起点：
- **初学者**: Chapter 1 → Chapter 2 → Chapter 3
- **有经验者**: Chapter 3 → Chapter 4 → Chapter 5
- **进阶用户**: Chapter 5 → Chapter 6 → Chapter 7

### 2. 环境准备
```bash
git clone https://github.com/your-repo/llm_sft.git
cd llm_sft

# 根据选择的章节安装依赖
pip install -r chapter_X/requirements.txt
```

### 3. 运行示例
```bash
# 以Chapter 3为例
cd chapter_3
python 1_generate_data.py    # 数据准备
python 4_train_all.py        # 开始训练
```

## 📊 实验成果展示

### 医疗推理能力展示
微调后的医疗模型具备专业级的推理能力：

```
Question: 医生，我最近胃部不适，听说有几种抗溃疡药物可以治疗，您能详细介绍一下这些药物的分类、作用机制以及它们是如何影响胃黏膜的保护与损伤平衡的吗？

模型回答: 当然可以。抗溃疡药物主要分为四类：抑酸药、胃黏膜保护剂、促胃动力药和抗幽门螺杆菌药物... [详细医学分析]
```

### 性能对比
| 微调方式 | 模型规模 | 显存占用 | 训练速度 | 推理质量 |
|---------|---------|---------|---------|---------|
| 全参数微调 | 1.7B | 12GB | 基准 | 优秀 |
| LoRA微调 | 1.7B | 6GB | 1.5x | 良好 |
| Unsloth+4bit | 7B | 8GB | 3x | 优秀 |

## 🤝 技术栈总览

### 训练框架
- **Transformers**: 核心训练框架，灵活可控
- **LLaMA-Factory**: 低代码训练，快速部署
- **Unsloth**: 内存优化，高效训练

### 模型家族
- **Qwen系列**: Qwen3-1.7B/8B, Qwen2.5-3B
- **DeepSeek**: DeepSeek-R1-Distill-Qwen-7B
- **MiniMind**: 轻量级训练框架

### 工具链
- **数据处理**: datasets, pandas, sentence-transformers
- **实验监控**: SwanLab, TensorBoard
- **模型评估**: EvalScope, ROUGE, BLEU
- **部署工具**: Llama.cpp, Ollama

## 📈 学习路线建议

### 阶段一：基础掌握（1-2周）
1. 完成Chapter 1-2的基础训练
2. 理解数据预处理和训练流程
3. 掌握基本的模型推理

### 阶段二：技能提升（2-3周）
1. 深入Chapter 3-4的微调技术
2. 对比不同训练框架的差异
3. 实践垂直领域适配

### 阶段三：高级应用（3-4周）
1. 掌握Chapter 5的大规模训练
2. 构建Chapter 6的评估体系
3. 完成Chapter 7的数据工程

### 阶段四：专业深化（持续学习）
1. 跟踪前沿技术发展
2. 参与开源项目贡献
3. 构建个人技术品牌

## 📞 技术支持

- **项目文档**: 详见各章节README
- **问题反馈**: 通过GitHub Issues提交
- **学习交流**: 欢迎提交PR分享经验

## 📄 许可证

本项目遵循MIT许可证，欢迎学习、使用和贡献。

---

> 🌟 **持续更新中**
> 本项目会持续跟进LLM训练领域的最新技术进展，欢迎关注和参与贡献！