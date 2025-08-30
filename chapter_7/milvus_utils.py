import json
import os
from typing import Any, Dict, List

import numpy as np
from dotenv import load_dotenv
from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      connections, db, utility)

from chapter_7.utils import model_embedding_bgem3


class MilvusHelper:
    """
    Milvus 2.5.6 常用功能封装
    """

    def __init__(self, db_name="", alias="medical"):
        load_dotenv()
        self.db_name = db_name if db_name else os.getenv('MILVUS_DATABASE_NAME')
        self.alias = alias  # 连接别名，用于后续操作关联
        self.collection = None

    def connect(self, timeout: int = 30):
        connections.connect(
            alias=self.alias,
            host=os.getenv('MILVUS_HOST'),
            port=os.getenv('MILVUS_PORT'),
            user=os.getenv('MILVUS_USER'),
            password=os.getenv('MILVUS_PASSWORD'),
            timeout=timeout
        )
        print(f"[MilvusHelper] Connected to Milvus successfully")

        # 切换或创建数据库，需指定连接别名
        if self.db_name not in db.list_database(using=self.alias):
            db.create_database(self.db_name, using=self.alias, timeout=timeout)
            print(f"[MilvusHelper] Database '{self.db_name}' created.")
        else:
            print(f"[MilvusHelper] Database '{self.db_name}' already exists.")
        db.using_database(self.db_name, using=self.alias)  # 切换到目标数据库

    def get_or_create_collection(self, collection_name: str):
        """
        创建（或获取已存在）的 Collection，支持可变长 varchar 字段
        """
        if utility.has_collection(collection_name, using=self.alias):
            self.collection = Collection(
                name=collection_name, using=self.alias)
            print(f"[MilvusHelper] Collection '{collection_name}' already exists.")
            return self.collection

        fields = [
            FieldSchema(name="id",
                        dtype=DataType.INT64,
                        is_primary=True,
                        auto_id=True),
            FieldSchema(name="instruction",
                        dtype=DataType.VARCHAR,
                        max_length=1024),
            FieldSchema(name="input",
                        dtype=DataType.VARCHAR,
                        max_length=1024),
            FieldSchema(name="output",
                        dtype=DataType.VARCHAR,
                        max_length=4096),
            FieldSchema(name="dense_vector",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=1024),
            FieldSchema(name="sparse_vector",
                        dtype=DataType.SPARSE_FLOAT_VECTOR)
        ]
        schema = CollectionSchema(
            fields=fields,
            description="医疗数据集",
            enable_dynamic_field=False
        )
        self.collection = Collection(name=collection_name, schema=schema, using=self.alias)
        print(f"[MilvusHelper] Collection '{collection_name}' created success.")
        return self.collection

    def drop_collection(self, collection_name: str):
        """删除 Collection"""
        if utility.has_collection(collection_name, using=self.alias):
            utility.drop_collection(collection_name, using=self.alias)
            print(
                f"[MilvusHelper] Collection '{collection_name}' dropped.")
        else:
            print(
                f"[MilvusHelper] Collection '{collection_name}' does not exist.")

    def build_index(self,
                    index_type: str = "IVF_FLAT",
                    metric_type: str = "COSINE",
                    nlist: int = 4096):
        """
        为向量字段创建索引
        """
        index_params = {
            "index_type": index_type,
            "metric_type": metric_type,
            "params": {"nlist": nlist}
        }
        self.collection.create_index(
            field_name="dense_vector",
            index_params=index_params
        )

        sparse_index_params = {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
            "params": {
                "inverted_index_algo": "DAAT_MAXSCORE",
            }
        }

        self.collection.create_index(
            field_name="sparse_vector",
            index_params=sparse_index_params
        )
        self.collection.load()
        print(f"[MilvusHelper] Index built on successful collection. "
              f"(type={index_type}, metric={metric_type}).")

    def insert(self, instructions: List[str], inputs: List[str], outputs: List[str], vectors_dict: dict,
               batch_size: int = 5000) -> List[int]:
        """
        批量插入数据，返回主键列表
        """
        vectors_list = vectors_dict.get("dense_vecs")
        sparse_list = vectors_dict.get("sparse_vecs")

        ids = []
        for i in range(0, len(vectors_list), batch_size):
            batch_instructions = instructions[i:i + batch_size]
            batch_inputs = inputs[i:i + batch_size]
            batch_outputs = outputs[i:i + batch_size]
            batch_vectors = vectors_list[i:i + batch_size]
            batch_sparse = sparse_list[i:i + batch_size]

            data = [
                batch_instructions,
                batch_inputs,
                batch_outputs,
                batch_vectors,
                batch_sparse
            ]
            insert_res = self.collection.insert(data)
            ids.extend(insert_res.primary_keys)
        return ids

    def flush(self):
        """强制刷盘"""
        self.collection.flush()
        print("[MilvusHelper] Flushed.")

    def search(self,
               query_vectors: np.ndarray,
               topk: int = 10,
               nprobe: int = 48,
               vector_field: str = "dense_vector",
               output_fields: List[str] = None) -> List[List[Dict[str, Any]]]:
        if output_fields is None:
            output_fields = ["instruction", "input", "output"]
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": nprobe}
        }
        res = self.collection.search(
            data=query_vectors.tolist(),
            anns_field=vector_field,
            param=search_params,
            limit=topk,
            output_fields=output_fields
        )
        # 整理格式
        results = []
        for hits in res:
            hits_list = []
            for hit in hits:
                item = hit.entity.to_dict()
                item["distance"] = hit.score
                hits_list.append(item)
            results.append(hits_list)
        return results

    def delete(self, expr: str):
        """
        根据表达式删除
        例：expr = "id in [1, 2, 3]"
        """
        self.collection.delete(expr=expr)
        print(f"[MilvusHelper] Deleted by expr: {expr}")

    def count(self) -> int:
        """返回 Collection 行数"""
        self.collection.flush()
        return self.collection.num_entities

    def release(self):
        """释放 Collection 内存"""
        self.collection.release()
        print("[MilvusHelper] Collection released.")

    def disconnect(self):
        """断开 Milvus"""
        connections.disconnect(self.alias)
        print("[MilvusHelper] Disconnected.")


def insert_data(json_path=r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\4_all_test.json"):
    helper = MilvusHelper()
    helper.connect()
    helper.get_or_create_collection("demo_test")
    helper.build_index()
    print("Milvus初始化完成，开始插入数据")

    with open(json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    instruction_list = []
    input_list = []
    output_list = []
    instruction_input_list = []

    for data in data_list:
        instruction_list.append(data["instruction"])
        input_list.append(data["input"])
        output_list.append(data["output"])
        instruction_input_list.append(data["instruction"] + data["input"])

    bge_dict = model_embedding_bgem3(instruction_input_list)

    helper.insert(instruction_list, input_list, output_list, bge_dict, batch_size=5)
    helper.flush()
    print("Total rows:", helper.count())


def search_data(search_query="头发越来越少，只掉不长，不知是心里抑郁造成"):
    helper = MilvusHelper()
    helper.connect()
    helper.get_or_create_collection("demo_test")

    bge_dict = model_embedding_bgem3(search_query)
    vectors_list = bge_dict.get("dense_vecs")
    result_data = helper.search(vectors_list)
    print(result_data)


if __name__ == "__main__":
    search_data()
