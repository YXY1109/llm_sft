import json
from typing import Dict, List

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def duplicate_removal(df_data, col_name):
    """
    基于字段去重
    :param df_data:
    :param col_name:
    :return:
    """
    df_data = df_data[df_data[col_name].notnull()]
    df_data = df_data.drop_duplicates(subset=[col_name])
    print(f"基于{col_name}字段去重后，数据条数：{len(df_data)}")
    return df_data


def build_corpus(records: List[Dict]) -> List[str]:
    """把 instruction 与 input 拼接成待编码的文本"""
    return [f"{r['instruction']} {r['input']}" for r in records]


def save_json(data: List[Dict], path: str, *, indent=2):
    """写回 JSON"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    # print(f"语义去重已保存去重后的结果至: {path}")


def semantic_deduplicate(
        file_input: str,
        file_output: str,
        threshold: float = 0.9,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
) -> str:
    """
    基于语义相似度去重
    :param file_input: 原始文件路径
    :param file_output: 输出文件路径
    :param threshold: 相似度阈值，超过即认为重复
    :param model_name: SentenceTransformer 模型
    :return: 去重后的数据文件
    """

    with open(file_input, "r", encoding="utf-8") as f:
        record_list = json.load(f)

    if not record_list:
        return record_list

    corpus = build_corpus(record_list)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(corpus, normalize_embeddings=True)

    sim_matrix = cosine_similarity(embeddings)  # shape=(n, n)

    keep_indices = []
    skip = set()
    n = len(record_list)

    for i in range(n):
        if i in skip:
            continue
        keep_indices.append(i)
        for j in range(i + 1, n):
            if j in skip:
                continue
            if sim_matrix[i, j] >= threshold:
                skip.add(j)

    deduped = [record_list[i] for i in keep_indices]
    print(f"原始条数: {n}, 去重后条数: {len(deduped)}")
    save_json(deduped, file_output)
    return file_output


def semantic_main():
    input_path = r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\4_all_test.json"  # 输入 JSON 文件
    out_path = r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\4_all_test_semantic.json"  # 输出 JSON 文件

    out_path = semantic_deduplicate(input_path, out_path)
    print(out_path)


if __name__ == "__main__":
    semantic_main()
