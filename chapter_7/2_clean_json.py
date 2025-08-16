import json
from typing import List


def merge_json_data(file_paths: List[str]) -> List[dict]:
    """
    从多个JSON文件路径读取数据，转换格式并合并为一个列表

    支持三种输入格式，统一转换为包含instruction、input、output的格式

    参数:
        file_paths: 包含多个JSON文件绝对路径的列表

    返回:
        合并并转换后的JSON列表
    """
    merged_data = []
    format1_count = 0  # Question, Complex_CoT, Response 格式
    format2_count = 0  # conversation 格式
    format3_count = 0  # instruction, input, output, history 格式
    error_count = 0  # 无法识别的格式

    def is_format1(item: dict) -> bool:
        """判断是否为第一种格式"""
        return all(key in item for key in ["Question", "Response"])

    def is_format2(item: dict) -> bool:
        """判断是否为第二种格式"""
        return "conversation" in item and isinstance(item.get("conversation"), list) and len(item["conversation"]) > 0

    def is_format3(item: dict) -> bool:
        """判断是否为第三种格式"""
        return all(key in item for key in ["instruction", "input", "output"])

    # 遍历每个文件路径
    for path in file_paths:
        try:
            # 读取单个JSON文件内容
            with open(path, 'r', encoding='utf-8') as f:
                json_content = json.load(f)

                # 确保读取的是列表类型
                if not isinstance(json_content, list):
                    print(f"警告: 文件 {path} 内容不是列表类型，已跳过")
                    continue

                file_added = 0  # 记录当前文件新增的数据量

                # 遍历并转换每个条目
                for item in json_content:
                    # 确保每个条目是字典类型
                    if not isinstance(item, dict):
                        print(f"警告: 文件 {path} 中包含非字典元素，已跳过")
                        error_count += 1
                        continue

                    converted = None

                    # 处理第一种格式
                    if is_format1(item):
                        converted = {
                            "instruction": "",
                            "input": item.get("Question", ""),
                            "output": item.get("Response", "")
                        }
                        format1_count += 1

                    # 处理第二种格式
                    elif is_format2(item):
                        # 取第一个对话作为内容
                        first_turn = item["conversation"][0]
                        converted = {
                            "instruction": first_turn.get("system", ""),
                            "input": first_turn.get("input", ""),
                            "output": first_turn.get("output", "")
                        }
                        format2_count += 1

                    # 处理第三种格式
                    elif is_format3(item):
                        converted = {
                            "instruction": item.get("instruction", ""),
                            "input": item.get("input", ""),
                            "output": item.get("output", "")
                        }
                        format3_count += 1

                    # 无法识别的格式
                    else:
                        print(f"警告: 文件 {path} 中包含无法识别的格式，已跳过")
                        error_count += 1
                        continue

                    if converted:
                        merged_data.append(converted)
                        file_added += 1

                print(f"成功处理文件: {path}，新增 {file_added} 条数据")

        except FileNotFoundError:
            print(f"错误: 找不到文件 {path}")
        except json.JSONDecodeError:
            print(f"错误: 文件 {path} 不是有效的JSON格式")
        except Exception as e:
            print(f"处理文件 {path} 时发生错误: {str(e)}")

    # 输出统计信息
    print("\n数据合并完成:")
    print(f"总计合并 {len(merged_data)} 条有效数据")
    print(f"格式1(Question-Response): {format1_count} 条")
    print(f"格式2(conversation): {format2_count} 条")
    print(f"格式3(instruction-input-output): {format3_count} 条")
    print(f"错误/无法识别格式: {error_count} 条")

    return merged_data


if __name__ == '__main__':
    json_path = [r"D:\PycharmProjects\llm_sft\chapter_7\data_json\medical_o1_sft.json",
                 r"D:\PycharmProjects\llm_sft\chapter_7\data_json\medical_o1_sft_Chinese.json",
                 r"D:\PycharmProjects\llm_sft\chapter_7\data_json\output2.jsonl",
                 r"D:\PycharmProjects\llm_sft\chapter_7\data_json\train_0001_of_0001.json"
                 ]
    data = merge_json_data(json_path)

    # 可以将合并后的数据保存到新的JSON文件
    with open('merge_data/2_medical.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("合并后的数据已保存到 2_medical.json")
