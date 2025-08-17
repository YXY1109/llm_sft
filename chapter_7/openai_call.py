import asyncio
import json
from typing import Any, Dict, List

import aiohttp

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
        "temperature": 0.0
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
            # resp.raise_for_status()
            data = await resp.json()
            print(f"[DEBUG] 打分结果：{data}")
            score_str = data["choices"][0]["message"]["content"].strip()
            score = float(score_str)
            if not (0.0 <= score <= 1.0):
                raise ValueError("Score out of range")
            return score
    except Exception as e:
        print(f"[WARN] 打分失败，返回 0.0：{e}")
        return 0.0


async def evaluate_batch(
        data: List[Dict[str, Any]],
        api_base: str,
        api_key: str,
        model: str,
        threshold: float,
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
                return {**item, "_score": score}

        tasks = [_score_with_sem(item, i + 1) for i, item in enumerate(data)]
        results = await asyncio.gather(*tasks)

    # 过滤
    cleaned = [r for r in results if r.pop("_score") >= threshold]
    return cleaned


async def async_main(input_file: str, output_file: str, threshold: float = 0.8):
    # 读取
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = await evaluate_batch(
        data,
        api_base="http://127.0.0.1:11434/v1",
        api_key="",
        model="qwen2.5:7b",
        threshold=threshold,
        max_concurrency=100,
    )

    # 写回
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"完成！保留 {len(cleaned)}/{len(data)} 条，阈值={threshold}")


def main():
    input_path = r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\4_all_test.json"
    # input_path = r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\4_all.json"
    output_path = r"D:\PycharmProjects\llm_sft\chapter_7\merge_data\8_all_test.json"

    asyncio.run(async_main(input_path, output_path))


if __name__ == "__main__":
    main()
