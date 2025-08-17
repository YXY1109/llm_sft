import asyncio

import pandas as pd

from chapter_7.openai_call import async_main
from chapter_7.utils import duplicate_removal, semantic_deduplicate

# json_data_path = r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\4_all.json" #完整数据
json_data_path = r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\4_all_test.json"  # 少量数据测试

# 1，转为df
df = pd.read_json(json_data_path)
print(f"1，原始数据条数：{len(df)}")  # 原始数据条数：1653894

# 2，数据instruction字段如果为空，保留，不为空的需要去重
df = duplicate_removal(df, "instruction")  # 基于instruction字段去重后，数据条数：1408392

# 3，input字段如果为空，数据保留，不为空的需要去重
df = duplicate_removal(df, "input")  # 基于input字段去重后，数据条数：611360

# 4，output字段如果为空，去除，需要去重
df_clean = df.dropna(subset=['output'])  # 去除NaN
df_clean = df_clean[df_clean['output'].astype(str).str.strip() != '']  # 去除空字符串（含仅空格的情况）
df = duplicate_removal(df_clean, "output")  # 基于output字段去重后，数据条数：584312

# 5，instruction+input字段如果为空，去除，需要去重
df_clean = df.dropna(subset=['instruction', 'input'], how='any')  # 先处理NaN值
# 再处理空字符串（包括仅含空格的情况）
df_clean = df_clean[(df_clean['instruction'].astype(str).str.strip() != '') &
                    (df_clean['input'].astype(str).str.strip() != '')]

# 6，基于instruction和input两个字段进行去重，保留首次出现的行
df = df_clean.drop_duplicates(subset=['instruction', 'input'], keep='first')
print(f"6，pf去重后，数据条数：{len(df)}")  # 去重后，数据条数：584308

save_pd_json = r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\6_all_pd.json"
# 将df保存为json
df.to_json(save_pd_json, orient="records", force_ascii=False)

# 7，instruction+input，语义去重（阈值 0.9）
out_path = r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\7_all_pd_deduplicate.json"
out_path_7 = semantic_deduplicate(save_pd_json, out_path)
print(f"7，基于语义去重后，数据路径：{out_path_7}")

# 8，使用LLM对回答质量打分
# input_path = r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\4_all_test.json"
output_path = r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\8_all_score.json"
asyncio.run(async_main(out_path_7, output_path))
print(f"8，基于LLM打分后，数据路径：{output_path}")

# 9，根据评分进行数据筛选
pass
