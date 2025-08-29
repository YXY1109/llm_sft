import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


def duplicate_removal(df_data, col_name):
    """
    基于字段去重
    :param df_data:
    :param col_name:
    :return:
    """
    # 空值掩码（NaN、None、空字符串、仅空白字符都算空）
    empty_mask = (
            df_data[col_name].isna()
            | df_data[col_name].astype(str).str.strip().eq('')
    )

    # 非空行去重，空行保持原样
    res = (
        pd.concat([
            df_data.loc[~empty_mask].drop_duplicates(subset=[col_name]),
            df_data.loc[empty_mask]
        ])
        .reset_index(drop=True)  # 如需保持连续索引
    )
    return res


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
   基于语义相似度去重，过时了，太慢了
   :param file_input: 原始文件路径
   :param file_output: 输出文件路径
   :param threshold: 相似度阈值，超过即认为重复
   :param model_name: SentenceTransformer 模型
   :param batch_size: 批处理大小
   :return: 去重后的数据文件
   """
    from sentence_transformers import SentenceTransformer

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
    print(f"语义去重已计算句向量，数据条数：{len(record_list)}")
    print(f"语义去重阈值为：{threshold}")

    # 3. 逐条去重
    keep_indices = []
    kept_vecs = []  # 仅保存已保留样本的向量
    # for idx, vec in enumerate(embeddings):
    for idx, vec in tqdm(enumerate(embeddings), total=len(embeddings), desc="语义去重"):
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


def df_score_null(file_input: str, threshold: float = 0.8):
    with open(file_input, encoding="utf-8") as f:
        record_list = json.load(f)
    # 过滤 _score >= 0.8
    print(f"LLM分数阈值为：{threshold}")
    filtered = [item for item in record_list if item.get("_score", 0) >= threshold]

    # 把 None 替换为 ""
    cleaned = [
        {k: (v if v is not None else "") for k, v in record.items()}
        for record in filtered
    ]
    return cleaned


def model_embedding_bgem3(input_words: List[str] | str) -> Dict[str, Any]:
    if isinstance(input_words, str):
        input_words = [input_words]

    input_words = [s.strip() for s in input_words if s.strip()]
    if not input_words:
        return {"dense_vecs": [], "sparse_vecs": [], "colbert_vecs": []}

    try:
        embedding_url = "http://127.0.0.1:8010/bge_m3"
        payload = {
            "sentences": input_words,
            "dense": True,
            "sparse": True,
            "colbert_vecs": False
        }
        timeout = max(10, len(input_words))
        resp = requests.post(embedding_url, json=payload, timeout=timeout)
        resp.raise_for_status()

        data_result = resp.json()["result"]

        return_dict = {
            "dense_vecs": np.array(data_result["dense_vecs"], dtype=np.float32),
            "dense_shape": data_result["dense_shape"],
            "sparse_vecs": [{int(k): float(v) for k, v in sp.items()}
                            for sp in data_result["sparse_vecs"]]
        }
        return return_dict
    except Exception as e:
        print(f"向量接口异常：{e}")
        return {}


def semantic_main():
    input_path = r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\4_all_test.json"  # 输入 JSON 文件
    out_path = r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\4_all_test_semantic.json"  # 输出 JSON 文件

    out_path = semantic_deduplicate_stream(input_path, out_path, batch_size=512)
    print(out_path)


if __name__ == "__main__":
    # semantic_main()

    # json_path = r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\8_all_score.json"
    # data_1 = df_score_null(json_path)
    # print(data_1)

    data = model_embedding_bgem3("你好")
    print(data)
