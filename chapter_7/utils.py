import json
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer


def duplicate_removal(df_data, col_name):
    """
    基于字段去重
    :param df_data:
    :param col_name:
    :return:
    """
    df_data = df_data[df_data[col_name].notnull()]
    df_data = df_data.drop_duplicates(subset=[col_name])
    # print(f"基于{col_name}字段去重后，数据条数：{len(df_data)}")
    return df_data


def build_corpus(records: List[Dict]) -> List[str]:
    """把 instruction 与 input 拼接成待编码的文本"""
    return [f"{r['instruction']} {r['input']}" for r in records]


def save_json(data: List[Dict], path: str, *, indent=2):
    """写回 JSON"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    # print(f"语义去重已保存去重后的结果至: {path}")


def semantic_deduplicate_stream(
        file_input: str,
        file_output: str,
        threshold: float = 0.9,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        batch_size: int = 64,
) -> str:
    """
       基于语义相似度去重
       :param file_input: 原始文件路径
       :param file_output: 输出文件路径
       :param threshold: 相似度阈值，超过即认为重复
       :param model_name: SentenceTransformer 模型
       :param batch_size: 批处理大小
       :return: 去重后的数据文件
       """

    # 1. 读数据
    with open(file_input, encoding="utf-8") as f:
        record_list = json.load(f)
    if not record_list:
        return record_list

    # 2. 计算句向量
    model = SentenceTransformer(model_name)
    corpus = [f"{r['instruction']} {r['input']}" for r in record_list]
    embeddings = model.encode(
        corpus, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=True
    )

    # 3. 逐条去重
    keep_indices = []
    kept_vecs = []  # 仅保存已保留样本的向量
    for idx, vec in enumerate(embeddings):
        if kept_vecs:  # 与之前所有保留样本比较
            sims = np.dot(kept_vecs, vec)  # shape=(k,)
            if sims.max() >= threshold:
                continue
        keep_indices.append(idx)
        kept_vecs.append(vec)

    # 4. 写回
    deduped = [record_list[i] for i in keep_indices]
    print(f"原始条数: {len(record_list)}, 去重后条数: {len(deduped)}")
    with open(file_output, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)
    return file_output


def semantic_main():
    input_path = r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\4_all_test.json"  # 输入 JSON 文件
    out_path = r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\4_all_test_semantic.json"  # 输出 JSON 文件

    out_path = semantic_deduplicate_stream(input_path, out_path)
    print(out_path)


if __name__ == "__main__":
    semantic_main()
