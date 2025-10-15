# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive LLM (Large Language Model) training and fine-tuning learning repository organized by chapters, each demonstrating different approaches and frameworks for training language models. The project covers everything from basic model training to advanced fine-tuning techniques, data processing, and model evaluation.

## Project Structure

The repository is organized into chapters, each focusing on different aspects of LLM training:

- **chapter_1**: From-scratch model training using MiniMind on AutoDL
- **chapter_2**: Dataset construction and Llama-Factory fine-tuning
- **chapter_3**: Hand-written implementation of Qwen3 fine-tuning for medical R1 reasoning
- **chapter_4**: Llama-Factory fine-tuning of Qwen2.5-3B-Instruct with financial data
- **chapter_5**: Unsloth-based fine-tuning of Qwen3-8B and DeepSeek-R1-Distill-Qwen-7B
- **chapter_6**: Model evaluation using EvalScope
- **chapter_7**: Medical dataset collection and cleaning
- **chapter_8**: Transformers library learning

## Common Development Commands

### Environment Setup

Each chapter may have its own requirements file. Install dependencies with:

```bash
# For chapter 3 (transformers-based fine-tuning)
pip install -r chapter_3/requirements.txt

# For chapter 5 (Unsloth-based fine-tuning)
pip install -r chapter_5/requirements.txt

# For chapter 7 (data cleaning)
pip install -r chapter_7/requirements.txt
```

### Training Commands

#### Chapter 3 - Transformers Fine-tuning
```bash
cd chapter_3
python 4_train_all.py      # Full model fine-tuning
python 5_train_lora.py     # LoRA fine-tuning
python 6_inference_all.py  # Model inference
```

#### Chapter 4 - Llama-Factory
```bash
cd chapter_4
python 1_download_model.py
python 2_qwen25_3b_lora_infer.py
python 3_qwen25_3b_merge_infer.py
```

#### Chapter 5 - Unsloth
```bash
cd chapter_5
python 1_download_model.py
python 2.1_sft_deepseek.py    # DeepSeek fine-tuning
python 2.2_sft_qwen3.py        # Qwen3 fine-tuning
python 3.1_infer_deepseek.py  # DeepSeek inference
python 4.1_infer_qwen3.py      # Qwen3 inference
```

#### Chapter 6 - Model Evaluation
```bash
cd chapter_6
python 1_qwen2.5_0.5b.py      # Evaluate Qwen2.5-0.5B
python 2_qwen3_8b.py          # Evaluate Qwen3-8B
python 3.1_rouge_bleu.py      # ROUGE and BLEU evaluation
python 3.2_llm_eval.py        # LLM-based evaluation
```

#### Chapter 7 - Data Cleaning
```bash
cd chapter_7
python 1_clean_dir.py         # Directory cleaning
python 2_clean_json.py        # JSON cleaning
python 3_clean_json.py        # Advanced JSON cleaning
python 4_clean_all.py         # Complete cleaning pipeline
python 5.1_clean_data.py      # Data quality cleaning
```

### Model and Data Paths

- Models are typically stored in `chapter_X/models/` directories
- Training outputs go to `models/train_*` or similar directories
- Data files are often stored in `chapter_X/data_json/` or similar
- Check each chapter's specific Python files for exact path configurations

## Key Technologies and Frameworks

### Core Libraries
- **transformers**: Hugging Face Transformers library (chapters 3, 4, 5, 8)
- **datasets**: For data loading and processing (chapters 3, 5)
- **peft**: Parameter-Efficient Fine-Tuning (chapter 3)
- **trl**: Transformer Reinforcement Learning (chapter 5)
- **unsloth**: Fast fine-tuning library (chapter 5)

### Fine-tuning Frameworks
- **LLama-Factory**: Low-code fine-tuning framework (chapter 4)
- **Unsloth**: Optimized fine-tuning with memory efficiency (chapter 5)
- **Hand-written**: Custom training loops (chapter 3)

### Evaluation
- **EvalScope**: ModelScope evaluation framework (chapter 6)
- **ROUGE/BLEU**: Text similarity metrics
- **LLM-based evaluation**: Using models to evaluate outputs

### Data Processing
- **Semantic deduplication**: Using sentence transformers (chapter 7)
- **Milvus**: Vector database for similarity search (chapter 7)
- **Data cleaning pipelines**: Comprehensive data quality tools

## Training Configurations

### Common Training Parameters
- **Max sequence length**: 2048-4096 tokens depending on model
- **Batch size**: 1-4 with gradient accumulation
- **Learning rates**: 1e-4 to 2e-4
- **LoRA rank**: Typically 8
- **LoRA alpha**: Typically 16-32

### Logging and Monitoring
- **SwanLab**: Experiment tracking (used in chapters 3, 4, 5)
- **ModelScope**: Model hub and dataset management

## Data Formats

### Training Data Format
Most chapters expect data in this format:
```json
{
  "instruction": "System prompt or instruction",
  "input": "User input/question",
  "output": "Expected response"
}
```

### Chapter 3 Medical Data
```json
{
  "question": "Medical question",
  "think": "Chain of thought reasoning",
  "answer": "Final answer"
}
```

## Model Types and Sizes

- **Qwen3-1.7B**: Small medical fine-tuning model
- **Qwen2.5-3B-Instruct**: Financial data fine-tuning
- **Qwen3-8B**: Large medical reasoning model
- **DeepSeek-R1-Distill-Qwen-7B**: Medical CoT reasoning
- **MiniMind**: Custom training from scratch (chapter 1)

## Development Notes

### GPU Requirements
- Most fine-tuning requires GPU with 16GB+ VRAM
- 4-bit quantization available in Unsloth for memory efficiency
- Gradient checkpointing is used to reduce memory usage

### Common Issues
- Memory errors: Reduce batch size or enable gradient checkpointing
- Model loading: Ensure correct model paths and trust_remote_code=True
- Data formatting: Check that data matches expected format for each chapter

### Integration with ModelScope
Many models and datasets are loaded from ModelScope hub. Ensure proper authentication and network access.

## Key File Locations

- Main training scripts: `chapter_*/[0-9]*.py`
- Requirements files: `chapter_*/requirements.txt`
- Model checkpoints: `chapter_*/models/` or output directories
- Data files: `chapter_*/data_json/` or chapter-specific directories
- Configuration files: Various `.yaml` files in output directories

This repository serves as a comprehensive learning resource for LLM fine-tuning techniques, with practical examples covering multiple frameworks and approaches.