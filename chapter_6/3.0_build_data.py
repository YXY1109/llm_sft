import json
import os
import random


def convert_medical_data(input_path, output_path, sample_size=100, random_seed=42):
    """
    从输入的JSONL文件中抽取指定数量的数据，转换为医疗评测格式并保存

    参数:
    input_path (str): 输入JSONL文件路径（原数据文件）
    output_path (str): 输出JSONL文件路径（转换后文件）
    sample_size (int): 抽取的数据条数，默认100
    random_seed (int): 随机种子，确保结果可复现，默认42
    """
    # 读取原始数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]

    # 检查数据量是否充足
    if len(data) < sample_size:
        raise ValueError(f"原始数据条数（{len(data)}）不足{sample_size}条，请检查输入文件")

    # 设置随机种子，保证抽取结果可复现
    random.seed(random_seed)
    sampled_data = random.sample(data, sample_size)

    # 转换格式并写入输出文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sampled_data:
            # 转换为医疗评测格式，优化system提示词
            converted = {
                "system": "你是一位专业的医疗答问专家，擅长解答各类健康医疗问题，提供准确、易懂的专业建议。",
                "query": item["question"],  # 对应原数据的question字段
                "response": item["answer"]  # 对应原数据的answer字段
            }
            # 每条数据写入一行，保持JSONL格式
            f.write(json.dumps(converted, ensure_ascii=False) + '\n')

    print(f"转换完成！已抽取{sample_size}条数据，保存至{output_path}")


if __name__ == '__main__':
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    original_data = os.path.join(parent_dir, "json_data", "r1_data_example.jsonl")
    new_data = os.path.join(parent_dir, "json_data", "r1_data_example_100.jsonl")
    print(original_data)
    print(new_data)
    convert_medical_data(
        input_path=original_data,
        output_path=new_data,
        sample_size=100,
        random_seed=666  # 可自定义种子，如100、2023等
    )
