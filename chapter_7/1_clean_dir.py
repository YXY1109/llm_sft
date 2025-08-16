import json
import os


def merge_dir_data(parent_dir):
    """
    合并父目录下所有直接子目录中的数据集
    处理每行一个JSON对象的文件格式
    :param parent_dir: 父目录路径
    :return: 合并后的数据集列表
    """
    merged_data = []

    # 获取父目录下的所有直接子目录
    for item in os.listdir(parent_dir):
        item_path = os.path.join(parent_dir, item)
        # 只处理目录，不处理文件
        if os.path.isdir(item_path):
            # 处理子目录中的所有文件
            for file in os.listdir(item_path):
                # 只处理JSON文件
                if file.endswith('.json'):
                    file_path = os.path.join(item_path, file)
                    try:
                        # 打开并读取JSON文件（每行一个JSON对象）
                        with open(file_path, 'r', encoding='utf-8') as f:
                            line_number = 0
                            for line in f:
                                line_number += 1
                                line = line.strip()
                                if not line:  # 跳过空行
                                    continue

                                try:
                                    data = json.loads(line)
                                    # 检查数据结构是否符合预期
                                    if isinstance(data, dict) and all(
                                            key in data for key in ['instruction', 'input', 'output']):
                                        merged_data.append(data)
                                    else:
                                        print(f"警告: {file_path} 第{line_number}行格式不符合要求，已跳过")
                                except json.JSONDecodeError:
                                    print(f"错误: 无法解析 {file_path} 第{line_number}行")
                    except Exception as e:
                        print(f"处理文件 {file_path} 时出错: {str(e)}")

    return merged_data


if __name__ == "__main__":
    # 重复值 / 缺失值 / 异常值处理、隐私脱敏、文本标准化
    parent_directory = r"D:\PycharmProjects\llm_sft\chapter_7\data_json"  # 替换为实际的父目录路径
    merged = merge_dir_data(parent_directory)
    print(f"合并完成，共合并了 {len(merged)} 条数据")

    # 可以将合并后的数据保存到新的JSON文件
    with open('merge_data/1_medical.json', 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print("合并后的数据已保存到 1_medical.json")
