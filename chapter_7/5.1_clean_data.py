import asyncio
import logging
import time
from contextlib import contextmanager

import pandas as pd

from chapter_7.openai_call import async_main
from chapter_7.utils import duplicate_removal, semantic_deduplicate_stream

# ====== 日志配置 ======
logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO
)


# ====== 计时器工具 ======
@contextmanager
def timer(step_name: str):
    """统计代码块耗时并打印日志"""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logging.info(f"{step_name} 耗时: {elapsed:.2f} 秒")


# 1，加载数据
json_data_path = r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\4_all.json"  # 完整数据
# json_data_path = r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\4_all_test.json"  # 少量数据测试

with timer("1，读取原始数据"):
    df = pd.read_json(json_data_path)
logging.info(f"1，原始数据条数：{len(df)}")

# 2，基于 instruction 去重
with timer("2，基于 instruction 去重"):
    df = duplicate_removal(df, "instruction")
logging.info(f"2，去重后数据条数：{len(df)}")

# 3，基于 input 去重
with timer("3，基于 input 去重"):
    df = duplicate_removal(df, "input")
logging.info(f"3，去重后数据条数：{len(df)}")

# 4，去除 output 为空的数据，再基于 output 去重
with timer("4，清理 output 并去重"):
    df_clean = df.dropna(subset=['output'])
    df_clean = df_clean[df_clean['output'].astype(str).str.strip() != '']
    df = duplicate_removal(df_clean, "output")
logging.info(f"4，去重后数据条数：{len(df)}")

# 5，清理 instruction 和 input 为空的数据
with timer("5，清理 instruction 和 input 为空的数据"):
    df_clean = df.dropna(subset=['instruction', 'input'], how='any')
    df_clean = df_clean[
        (df_clean['instruction'].astype(str).str.strip() != '') &
        (df_clean['input'].astype(str).str.strip() != '')
        ]

# 6，基于 instruction+input 去重
with timer("6，基于 instruction+input 去重"):
    df = df_clean.drop_duplicates(subset=['instruction', 'input'], keep='first')
logging.info(f"6，去重后数据条数：{len(df)}")

# 保存中间结果
save_pd_json = r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\6_all_pd.json"
with timer("6.1，保存中间 JSON"):
    df.to_json(save_pd_json, orient="records", force_ascii=False)

# 7，语义去重，本地执行
out_path = r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\7_all_pd_deduplicate.json"
with timer("7，语义去重"):
    out_path_7 = semantic_deduplicate_stream(save_pd_json, out_path)
logging.info(f"7，语义去重后数据路径：{out_path_7}")

# 8，LLM 打分，百炼接口，注意token消耗
output_path = r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\8_all_score.json"
with timer("8，LLM 打分"):
    asyncio.run(async_main(out_path_7, output_path))
logging.info(f"8，打分后数据路径：{output_path}")

# 9，后续筛选
logging.info("全部步骤完成，可继续进行第 9 步筛选")
