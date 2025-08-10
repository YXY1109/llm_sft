import json
import random
import string
import time

import orjson
import ujson


def random_string(length=1000):
    """Generate a random string of fixed length"""
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


data = {
    "name": random_string(1000),
    "array": list(range(1000000)),
    "nested": {
        "subarray": list(range(10000)),
        "value": random_string(10000)
    }
}

# 确保所有 json 库都可以对同一份数据进行序列化和反序列化
assert data == json.loads(json.dumps(data))
assert data == ujson.loads(ujson.dumps(data))
assert data == orjson.loads(orjson.dumps(data))

start_time = time.time()
json.dumps(data)
json_dump_time = time.time() - start_time

start_time = time.time()
json.loads(json.dumps(data))
json_load_time = time.time() - start_time

start_time = time.time()
ujson.dumps(data)
ujson_dump_time = time.time() - start_time

start_time = time.time()
ujson.loads(ujson.dumps(data))
ujson_load_time = time.time() - start_time

start_time = time.time()
orjson.dumps(data)
orjson_dump_time = time.time() - start_time

start_time = time.time()
orjson.loads(orjson.dumps(data))
orjson_load_time = time.time() - start_time

print(f"json dump: {json_dump_time:.5f}s, load: {json_load_time:.5f}s")
print(f"ujson dump: {ujson_dump_time:.5f}s, load: {ujson_load_time:.5f}s")
print(f"orjson dump: {orjson_dump_time:.5f}s, load: {orjson_load_time:.5f}s")
