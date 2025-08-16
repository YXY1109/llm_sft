import json


def merge_jsonl_data(file_path):
    """
    读取JSONL文件并将其转换为指定格式
    原始格式字段: instruction, question, think, answer, metrics
    目标格式字段: instruction, input, output
    """
    merged_data = []

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
                    # 解析JSON数据
                    data = json.loads(line)

                    # 检查原始数据是否包含必要字段
                    required_fields = ['instruction', 'question', 'answer']
                    if not isinstance(data, dict) or not all(field in data for field in required_fields):
                        print(f"警告: {file_path} 第{line_number}行缺少必要字段，已跳过")
                        continue

                    # 按照指定格式映射字段
                    transformed = {
                        'instruction': data['instruction'],
                        'input': data['question'],
                        'output': data['answer']  # 合并think和answer作为output
                    }

                    merged_data.append(transformed)

                except json.JSONDecodeError as e:
                    print(f"错误: 无法解析 {file_path} 第{line_number}行 - {str(e)}")
                except Exception as e:
                    print(f"处理 {file_path} 第{line_number}行时出错 - {str(e)}")

    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
    except PermissionError:
        print(f"错误: 没有权限访问文件 {file_path}")
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")

    return merged_data


# 使用示例
if __name__ == "__main__":
    json_file = r"D:\PycharmProjects\llm_sft\chapter_7\data_json\r1_data_example.jsonl"
    merged = merge_jsonl_data(json_file)
    print(f"合并完成，共合并了 {len(merged)} 条数据")

    # 可以将合并后的数据保存到新的JSON文件
    with open('merge_data/3_medical.json', 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print("合并后的数据已保存到 3_medical.json")
