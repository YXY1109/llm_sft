import asyncio
import json
import os
from typing import Any, Dict, List

import aiohttp
from dotenv import load_dotenv

from chapter_7.prompt import DEFAULT_PROMPT_TEMPLATE_ZH


async def async_openai_score(
        session: aiohttp.ClientSession,
        instruction: str,
        user_input: str,
        output: str,
        api_base: str,
        api_key: str,
        model: str,
        timeout: int = 120,
) -> float:
    prompt = DEFAULT_PROMPT_TEMPLATE_ZH.format(
        instruction=instruction,
        input=user_input,
        output=output,
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.01
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with session.post(
                f"{api_base.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            # print(f"[DEBUG] 打分结果：{data}")
            score_str = data["choices"][0]["message"]["content"].strip()
            score = float(score_str)
            if 0.0 <= score <= 1.0:
                return score
            else:
                print(f"[WARN] 打分超出0到1，返回 0.0")
                return 0.0
    except Exception as e:
        print(f"[WARN] 打分失败，返回 0.0：{e}")
        return 0.0


async def evaluate_batch(
        data: List[Dict[str, Any]],
        api_base: str,
        api_key: str,
        model: str,
        max_concurrency: int = 50,
) -> List[Dict[str, Any]]:
    connector = aiohttp.TCPConnector(limit=max_concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(max_concurrency)

        async def _score_with_sem(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
            async with sem:
                score = await async_openai_score(
                    session,
                    item["instruction"],
                    item["input"],
                    item["output"],
                    api_base,
                    api_key,
                    model,
                )
                print(f"[{idx:>4}/{len(data)}] score={score:.3f}")
                # 直接把分数写回
                return {**item, "_score": score}

        tasks = [_score_with_sem(item, i + 1) for i, item in enumerate(data)]
        results = await asyncio.gather(*tasks)

    # 不再过滤，全部返回
    return results


async def async_main(input_file: str, output_file: str):
    load_dotenv()  # 加载.env文件

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = await evaluate_batch(
        data,
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        model="qwen2.5-72b-instruct",
        max_concurrency=100,
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"完成！共处理 {len(results)} 条数据")


def main():
    input_path = r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\4_all_test.json"
    output_path = r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\8_all_test.json"
    asyncio.run(async_main(input_path, output_path))


if __name__ == "__main__":
    main()
