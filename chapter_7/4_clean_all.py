import json


def merge_all_data(file_paths):
    """
    读取JSONL文件并将其转换为指定格式
    原始格式字段: instruction, question, think, answer, metrics
    目标格式字段: instruction, input, output
    """
    merged_data = []

    for file_path in file_paths:
        # 打开并读取JSON文件（每行一个JSON对象）
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 解析JSON数据
                json_content = json.load(f)

                for data in json_content:
                    # 按照指定格式映射字段
                    transformed = {
                        'instruction': data['instruction'],
                        'input': data['input'],
                        'output': data['output']
                    }
                    merged_data.append(transformed)
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")

    return merged_data


# 使用示例
if __name__ == "__main__":
    merge_list = [
        r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\1_medical.json",
        r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\2_medical.json",
        r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\3_medical.json",
    ]
    merged = merge_all_data(merge_list)
    print(f"合并完成，共合并了 {len(merged)} 条数据")

    # 可以将合并后的数据保存到新的JSON文件
    with open('merge_data/4_all.json', 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print("合并后的数据已保存到 4_all.json")
