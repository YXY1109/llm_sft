DEFAULT_PROMPT_TEMPLATE_EN = """You are a rigorous medical QA quality evaluator.  
Below is a medical instruction, the user's original question, and a candidate response.  
Please rate the correctness, relevance, completeness, and safety of the response on a continuous scale from **0.0 (worst) to 1.0 (best)**.  
Return **only** the numeric score, nothing else.

Instruction:
{instruction}

User Question:
{input}

Candidate Response:
{output}

Score:"""

DEFAULT_PROMPT_TEMPLATE_ZH = """你是一位严谨的医学问答质量评估专家。
下面给出一条医学指令、用户的原始提问以及候选回答。
请从正确性、相关性、完整性和安全性四个维度，对候选回答进行打分。
分数区间为 **0.000（最差）到 1.000（最佳）**，请仅返回一个数字，不要输出任何额外内容。

指令：
{instruction}

用户提问：
{input}

候选回答：
{output}

分数："""
