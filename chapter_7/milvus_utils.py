import os
from typing import List, Dict, Any, Optional

import numpy as np
from dotenv import load_dotenv
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType,
    Collection, utility, db
)


class MilvusHelper:
    """
    Milvus 2.5.6 常用功能封装
    """

    def __init__(self, alias: str = "medical", timeout: int = 30):
        """
        连接 Milvus Server
        """
        load_dotenv()
        self.alias = alias
        self.db_name = os.getenv('MILVUS_DATABASE_NAME')

        connections.connect(
            alias=self.alias,
            host=os.getenv('MILVUS_HOST'),
            port=os.getenv('MILVUS_PORT'),
            user=os.getenv('MILVUS_USER'),
            password=os.getenv('MILVUS_PASSWORD'),
            db_name=self.db_name,
            timeout=timeout
        )
        self.collection: Optional[Collection] = None
        print(f"[MilvusHelper] Connected to db={self.db_name}")

    # ----------------------------------
    # 数据库级别
    # ----------------------------------
    def create_database(self, timeout: int = 10):
        """若数据库不存在则创建"""
        if self.db_name not in db.list_database():
            db.create_database(self.db_name, timeout=timeout)
            print(f"[MilvusHelper] Database '{self.db_name}' created.")
        else:
            print(f"[MilvusHelper] Database '{self.db_name}' already exists.")

    def use_database(self):
        """切换数据库"""
        db.using_database(self.db_name)
        print(f"[MilvusHelper] Switched to database '{self.db_name}'.")

    # ----------------------------------
    # Collection 级别
    # ----------------------------------
    def create_collection(self, collection_name: str):
        """
        创建（或获取已存在）的 Collection，支持可变长 varchar 字段
        """
        if utility.has_collection(collection_name, using=self.alias):
            self.collection = Collection(
                name=collection_name, using=self.alias)
            print(
                f"[MilvusHelper] Collection '{collection_name}' already exists.")
            return self.collection

        fields = [
            FieldSchema(name="id",
                        dtype=DataType.INT64,
                        is_primary=True,
                        auto_id=True),
            FieldSchema(name="instruction",
                        dtype=DataType.VARCHAR,
                        max_length=512),
            FieldSchema(name="input",
                        dtype=DataType.VARCHAR,
                        max_length=512),
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

    # ----------------------------------
    # 索引
    # ----------------------------------
    def build_index(self,

                    index_type: str = "IVF_FLAT",
                    metric_type: str = "COSINE",
                    nlist: int = 8192):
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
        self.collection.load()  # 建完索引后 load
        print(
            f"[MilvusHelper] Index built on successful collection. "
            f"(type={index_type}, metric={metric_type}).")

    # ----------------------------------
    # 数据写入
    # ----------------------------------
    def insert(self,
               vectors: np.ndarray,
               texts: List[str],
               batch_size: int = 5000) -> List[int]:
        """
        批量插入数据，返回主键列表
        """

        ids = []
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i + batch_size].tolist()
            batch_texts = texts[i:i + batch_size]

            batch_sparse = []
            for row in batch_vectors:
                # 取绝对值最大的 topk 个维度
                indices = np.argpartition(np.abs(row), -50)[-50:]
                sparse_dict = {int(idx): float(row[idx])
                               for idx in indices if row[idx] != 0}
                batch_sparse.append(sparse_dict)

            data = [
                batch_texts,
                batch_texts,
                batch_texts,
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

    # ----------------------------------
    # 相似度检索
    # ----------------------------------
    def search(self,
               query_vectors: np.ndarray,
               topk: int = 10,
               nprobe: int = 16,
               vector_field: str = "dense_vector",
               output_fields: List[str] = None) -> List[List[Dict[str, Any]]]:
        """
        向量检索，返回 [[{"id":..., "distance":..., "text":...}, ...], ...]
        """
        if output_fields is None:
            output_fields = ["output"]
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

    # ----------------------------------
    # 查询 & 删除
    # ----------------------------------
    def query(self,
              expr: str,
              output_fields: List[str] = None,
              limit: int = 100) -> List[Dict[str, Any]]:
        """
        根据表达式查询
        例：expr = "id in [1, 2, 3]"
        """
        if output_fields is None:
            output_fields = ["id", "input"]
        res = self.collection.query(
            expr=expr,
            output_fields=output_fields,
            limit=limit,
            using=self.alias
        )
        return res

    def delete(self, expr: str):
        """
        根据表达式删除
        例：expr = "id in [1, 2, 3]"
        """
        self.collection.delete(expr=expr)
        print(f"[MilvusHelper] Deleted by expr: {expr}")

    # ----------------------------------
    # 统计信息
    # ----------------------------------
    def count(self) -> int:
        """返回 Collection 行数"""
        self.collection.flush()
        return self.collection.num_entities

    # ----------------------------------
    # 资源释放
    # ----------------------------------
    def release(self):
        """释放 Collection 内存"""
        self.collection.release()
        print("[MilvusHelper] Collection released.")

    def disconnect(self):
        """断开 Milvus"""
        connections.disconnect(self.alias)
        print("[MilvusHelper] Disconnected.")


# -------------------------------------------------
# 使用示例（可直接 python milvus_helper.py 运行测试）
# -------------------------------------------------
if __name__ == "__main__":
    milvus = MilvusHelper()
    milvus.create_collection("demo")
    milvus.build_index()

    # 构造 100 条随机数据
    vectors = np.random.rand(100, 1024).astype("float32")
    texts = [f"text_{i}" for i in range(100)]
    milvus.insert(vectors, texts)
    milvus.flush()
    print("Total rows:", milvus.count())

    # 检索
    q = np.random.rand(1, 1024).astype("float32")
    res = milvus.search(q, topk=5)
    print("Search result:", res)

    # 查询
    rows = milvus.query("id < 10")
    print("Query result:", rows)

    # 清理
    milvus.release()
    milvus.drop_collection("demo")
    milvus.disconnect()
