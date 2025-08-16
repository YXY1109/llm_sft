import pandas as pd


def duplicate_removal(df_data, col_name):
    df_data = df_data[df_data[col_name].notnull()]
    df_data = df_data.drop_duplicates(subset=[col_name])
    print(f"基于{col_name}字段去重后，数据条数：{len(df_data)}")
    return df_data


json_data_path = r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\4_all.json"
# json_data_path = r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\4_all_test.json"

# 转为df
df = pd.read_json(json_data_path)
print(f"原始数据条数：{len(df)}")  # 原始数据条数：1653894

# 数据instruction字段如果为空，保留，不为空的需要去重
df = duplicate_removal(df, "instruction")  # 基于instruction字段去重后，数据条数：1408392

# input字段如果为空，数据保留，不为空的需要去重
df = duplicate_removal(df, "input")  # 基于input字段去重后，数据条数：611360

# output字段如果为空，去除，需要去重
df_clean = df.dropna(subset=['output'])  # 去除NaN
df_clean = df_clean[df_clean['output'].astype(str).str.strip() != '']  # 去除空字符串（含仅空格的情况）
df = duplicate_removal(df_clean, "output")  # 基于output字段去重后，数据条数：584312

# instruction+input字段如果为空，去除，需要去重
# 先处理NaN值
df_clean = df.dropna(subset=['instruction', 'input'], how='any')
# 再处理空字符串（包括仅含空格的情况）
df_clean = df_clean[
    (df_clean['instruction'].astype(str).str.strip() != '') &
    (df_clean['input'].astype(str).str.strip() != '')
    ]
# 2. 基于instruction和input两个字段进行去重，保留首次出现的行
df = df_clean.drop_duplicates(subset=['instruction', 'input'], keep='first')
print(f"去重后，数据条数：{len(df)}")  # 去重后，数据条数：584308

# instruction+input，语义去重（阈值 0.8）

# output与instruction+input的语义关联性，合理性判断

# 统计instruction+input的长度，output的长度，如果特别长的，去掉

# 使用LLM对回答质量打分
