import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from dotenv import load_dotenv
from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      connections, db, utility)
from tqdm import tqdm

from chapter_7.utils import model_embedding_bgem3


class MilvusHelper:
    def __init__(self, db_name: str = "", alias: str = "medical"):
        load_dotenv()
        self.db_name = db_name if db_name else os.getenv('MILVUS_DATABASE_NAME', "default_db")
        self.alias = alias  # 连接别名，用于后续操作关联
        self.collection: Optional[Collection] = None
        self.connected: bool = False
        self._connect()

    def _connect(self, timeout: int = 30) -> None:
        """连接到Milvus并切换到指定数据库"""
        if self.connected:
            print(f"[MilvusHelper] Already connected to Milvus")
            return

        try:
            connections.connect(
                alias=self.alias,
                host=os.getenv('MILVUS_HOST', "localhost"),
                port=os.getenv('MILVUS_PORT', "19530"),
                user=os.getenv('MILVUS_USER'),
                password=os.getenv('MILVUS_PASSWORD'),
                timeout=timeout
            )
            self.connected = True
            print(f"[MilvusHelper] Connected to Milvus successfully")

            # 切换或创建数据库
            if self.db_name not in db.list_database(using=self.alias):
                db.create_database(self.db_name, using=self.alias, timeout=timeout)
                print(f"[MilvusHelper] Database '{self.db_name}' created.")
            else:
                print(f"[MilvusHelper] Database Database '{self.db_name}' already exists.")
            db.using_database(self.db_name, using=self.alias)  # 切换到目标数据库
        except Exception as e:
            print(f"[MilvusHelper] Failed to connect to Milvus: {str(e)}")
            raise

    def get_or_create_collection(self, collection_name: str) -> Collection:
        """创建（或获取已存在）的Collection"""
        if not self.connected:
            raise Exception("Not connected to Milvus. Call connect() first.")

        if utility.has_collection(collection_name, using=self.alias):
            self.collection = Collection(name=collection_name, using=self.alias)
            print(f"[MilvusHelper] Collection '{collection_name}' already exists.")
            return self.collection

        # 定义集合结构
        fields = [
            FieldSchema(name="index", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="instruction", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="input", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="output", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR)
        ]

        schema = CollectionSchema(
            fields=fields,
            description="医疗数据集",
            enable_dynamic_field=False
        )

        self.collection = Collection(name=collection_name, schema=schema, using=self.alias)
        print(f"[MilvusHelper] Collection '{collection_name}' created successfully.")
        return self.collection

    def build_index(self,
                    index_type: str = "IVF_FLAT",
                    metric_type: str = "COSINE",
                    nlist: int = 4096) -> None:
        """为向量字段创建索引"""
        if not self.collection:
            raise Exception("No collection selected. Call get_or_create_collection() first.")

        # 创建稠密向量索引
        dense_index_params = {
            "index_type": index_type,
            "metric_type": metric_type,
            "params": {"nlist": nlist}
        }
        self.collection.create_index(
            field_name="dense_vector",
            index_params=dense_index_params
        )

        # 创建稀疏向量索引
        sparse_index_params = {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
            "params": {"inverted_index_algo": "DAAT_MAXSCORE"}
        }
        self.collection.create_index(
            field_name="sparse_vector",
            index_params=sparse_index_params
        )

        self.collection.load()
        print(f"[MilvusHelper] Index built successfully (type={index_type}, metric={metric_type}).")

    def insert(self,
               index: List[int],
               instructions: List[str],
               inputs: List[str],
               outputs: List[str],
               vectors_dict: dict,
               batch_size: int = 5000) -> List[int]:
        """批量插入数据，返回主键列表"""
        if not self.collection:
            raise Exception("No collection selected. Call get_or_create_collection() first.")

        vectors_list = vectors_dict.get("dense_vecs")
        sparse_list = vectors_dict.get("sparse_vecs")

        if len(vectors_list) != len(sparse_list):
            raise ValueError("vectors_dict must contain both 'dense_vecs' and 'sparse_vecs'")

        if not all(len(lst) == len(index) for lst in [instructions, inputs, outputs, vectors_list, sparse_list]):
            raise ValueError("All input lists must have the same length")

        ids = []
        total_batches = (len(index) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(index), batch_size), desc="Inserting batches", total=total_batches):
            batch_id = index[i:i + batch_size]
            batch_instructions = instructions[i:i + batch_size]
            batch_inputs = inputs[i:i + batch_size]
            batch_outputs = outputs[i:i + batch_size]
            batch_vectors = vectors_list[i:i + batch_size]
            batch_sparse = sparse_list[i:i + batch_size]

            data = [
                batch_id,
                batch_instructions,
                batch_inputs,
                batch_outputs,
                batch_vectors,
                batch_sparse
            ]

            try:
                insert_res = self.collection.insert(data)
                ids.extend(insert_res.primary_keys)
            except Exception as e:
                print(f"[MilvusHelper] Error inserting batch {i // batch_size + 1}: {str(e)}")
                raise

        return ids

    def flush(self) -> None:
        """强制刷盘"""
        if self.collection:
            self.collection.flush()
            self.collection.load()
            print("[MilvusHelper] Data flushed to disk.")

    def search(self,
               query_vectors: np.ndarray,
               topk: int = 10,
               nprobe: int = 48,
               vector_field: str = "dense_vector",
               output_fields: Optional[List[str]] = None) -> List[List[Dict[str, Any]]]:
        """搜索相似向量"""
        if not self.collection:
            raise Exception("No collection selected. Call get_or_create_collection() first.")

        if output_fields is None:
            output_fields = ["instruction", "input", "output", "index"]

        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": nprobe}
        }

        try:
            res = self.collection.search(
                data=query_vectors.tolist(),
                anns_field=vector_field,
                param=search_params,
                limit=topk,
                output_fields=output_fields
            )
        except Exception as e:
            print(f"[MilvusHelper] Search error: {str(e)}")
            raise

        # 整理结果格式
        results = []
        for hits in res:
            hits_list = []
            for hit in hits:
                item = hit.entity.to_dict()
                item["distance"] = hit.score
                hits_list.append(item)
            results.append(hits_list)

        return results

    def delete(self, expr: str) -> None:
        """根据表达式删除数据"""
        if not self.collection:
            raise Exception("No collection selected. Call get_or_create_collection() first.")

        try:
            self.collection.delete(expr=expr)
            print(f"[MilvusHelper] Deleted records matching expression: {expr}")
        except Exception as e:
            print(f"[MilvusHelper] Error deleting records: {str(e)}")
            raise

    def count(self) -> int:
        """返回Collection中的记录数"""
        if not self.collection:
            raise Exception("No collection selected. Call get_or_create_collection() first.")

        self.flush()
        return self.collection.num_entities

    def release(self) -> None:
        """释放Collection内存"""
        if self.collection:
            self.collection.release()
            print("[MilvusHelper] Collection released from memory.")

    def disconnect(self) -> None:
        """断开Milvus连接"""
        if self.connected:
            connections.disconnect(self.alias)
            self.connected = False
            print("[MilvusHelper] Disconnected from Milvus.")

    def load_and_preprocess_data(self, json_path: str) -> Tuple[List[int], List[str], List[str], List[str], List[str]]:
        """加载并预处理JSON数据"""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data_list = json.load(f)
        except Exception as e:
            print(f"[MilvusHelper] Error loading data from {json_path}: {str(e)}")
            raise

        id_list = []
        instruction_list = []
        input_list = []
        output_list = []
        instruction_input_list = []

        for idx, data in enumerate(data_list, 1):
            id_list.append(idx)
            instruction_list.append(data.get("instruction", ""))
            input_list.append(data.get("input", ""))
            output_list.append(data.get("output", ""))
            instruction_input_list.append(f"{data.get('instruction', '')}{data.get('input', '')}")

        return id_list, instruction_list, input_list, output_list, instruction_input_list

    def insert_from_json(self, json_path: str, collection_name: str, batch_size: int = 5000) -> None:
        """从JSON文件插入数据到指定集合"""
        # 连接并初始化集合
        self.get_or_create_collection(collection_name)
        self.build_index()
        print("Milvus initialized, starting data insertion...")

        # 加载和预处理数据
        id_list, instruction_list, input_list, output_list, instruction_input_list = self.load_and_preprocess_data(
            json_path)

        # 生成嵌入向量
        print("Generating embeddings...")
        bge_dict = model_embedding_bgem3(instruction_input_list)

        # 插入数据
        print(f"Inserting {len(id_list)} records...")
        self.insert(
            id_list,
            instruction_list,
            input_list,
            output_list,
            bge_dict,
            batch_size=batch_size
        )

        self.flush()
        print(f"Insertion complete. Total rows: {self.count()}")

    def search_similar(self, query: str, topk: int = 10) -> List[List[Dict[str, Any]]]:
        """搜索与查询相似的记录"""
        if not self.collection:
            raise Exception("No collection selected. Call get_or_create_collection() first.")

        # 生成查询向量
        bge_dict = model_embedding_bgem3(query)
        vectors_list = bge_dict.get("dense_vecs")

        # 执行搜索
        return self.search(np.array(vectors_list), topk=topk)

    def remove_duplicates(self, json_path: str, collection_name: str, threshold: float = 0.9) -> Set[int]:
        """移除重复数据，返回被标记为重复的ID集合"""
        if not self.collection:
            self.get_or_create_collection(collection_name)

        # 加载数据
        _, _, _, _, instruction_input_list = self.load_and_preprocess_data(json_path)

        skipped_ids: Set[int] = set()

        # 检查重复数据
        for index, text in tqdm(enumerate(instruction_input_list, 1),
                                desc="Checking for duplicates",
                                total=len(instruction_input_list)):
            if index in skipped_ids:
                continue

            # 搜索相似数据
            try:
                milvus_results = self.search_similar(text, topk=10)

                # 跳过第一个结果（自身），检查其他结果
                for result in milvus_results[0][1:]:
                    distance = result.get('distance', 0)
                    similar_id = result.get('index')

                    if distance > threshold and similar_id not in skipped_ids:
                        skipped_ids.add(similar_id)
                        print(f"Marked as duplicate (distance: {distance:.2f}): ID {similar_id}")

            except Exception as e:
                print(f"Error checking duplicates for ID {index}: {str(e)}")
                continue

        print(f"Found {len(skipped_ids)} duplicate records")
        return skipped_ids


def main():
    # 配置
    json_path = Path(r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\4_all_test.json")
    collection_name = "demo_test"

    # 初始化MilvusHelper
    milvus_helper = MilvusHelper()

    try:
        # 插入数据
        milvus_helper.insert_from_json(str(json_path), collection_name, batch_size=5)
        print(f"Total rows: {milvus_helper.count()}")

        # 移除重复数据
        duplicate_ids = milvus_helper.remove_duplicates(str(json_path), collection_name)
        print(f"Skipped duplicate IDs: {duplicate_ids}")

        # 示例查询
        sample_query = "头发越来越少，只掉不长，不知是心里抑郁造成"
        print(f"\nSearching for: {sample_query}")
        results = milvus_helper.search_similar(sample_query, topk=5)
        print(f"Search results:{results}")

        # 打印前5个结果
        for i, result in enumerate(results[0][:5], 1):
            print(f"\nResult {i}:")
            print(f"Distance: {result['distance']:.4f}")
            print(f"Instruction: {result['entity']['instruction']}")
            print(f"Input: {result['entity']['input']}")
            print(f"Output: {result['entity']['output']}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # 释放资源并断开连接
        # milvus_helper.release()
        milvus_helper.disconnect()


if __name__ == "__main__":
    main()
