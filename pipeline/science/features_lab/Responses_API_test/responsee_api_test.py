#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()


SYSTEM_PROMPT = (
    "你是严谨的信息研究助理。需要实时或外部信息时先使用 web_search 工具；"
    "整合检索结果后输出清晰要点，并在关键结论后给出可点击的来源注释。"
    "如果证据不足，要明确说明不确定性并给出下一步可验证的线索。"
)

def build_tools(sites: Optional[List[str]], use_preview: bool):
    """
    返回 tools 配置：
    - 默认使用官方托管的 web_search
    - 若提供 --sites 白名单，则以预览配置形式传入（不同版本字段名略有变化，脚本做了兼容处理）
    """
    if sites:
        # 预览版/带参数的 Web Search（允许传入站点白名单、位置等）
        return [{
            "type": "web_search_preview",
            # 一些常用可选项（按需增减；不同版本 SDK 字段名可能略有差异）
            "sites": sites,                     # 仅搜索这些站点
            "search_context_size": "medium",    # 返回上下文规模：small/medium/large
            "user_location": {                  # 近似位置（用于本地化结果排序）
                "type": "approximate",
                "country": "US",
                "region": "CA",
                "city": "Berkeley"
            }
        }]
    else:
        # 最小可用：一行开启托管 Web Search
        return [{"type": "web_search"}]


def call_responses_once(
    client: OpenAI,
    query: str,
    model: str,
    sites: Optional[List[str]] = None,
    stream: bool = False,
):
    tools = build_tools(sites, use_preview=bool(sites))

    # 构造 Responses API 输入（messages/role 风格由 SDK 统一到 input）
    payload = dict(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        tools=tools,
        parallel_tool_calls=True,
        # 开启“可显示的推理摘要”（不会泄露原始 chain-of-thought）
        reasoning={"effort": "medium", "summary": "auto"},
    )

    if not stream:
        resp = client.responses.create(**payload)
        print("\n=== Final Answer ===\n")
        print(resp.output_text.strip())
        # 尝试附加“推理摘要”（不同模型/权限下可能为空）
        try:
            rs = getattr(resp, "reasoning", None)
            if rs and getattr(rs, "summary", None):
                print("\n--- Reasoning Summary ---\n" + rs.summary.strip())
        except Exception:
            pass
        return

    # 流式输出：逐步显示“开始推理/正在搜索/整合证据/最终答案…”
    print("• 开始推理…")
    try:
        with client.responses.stream(**payload) as stream_obj:
            for event in stream_obj:
                et = getattr(event, "type", "")
                # 这些事件名以官方 SDK 为准；不同版本可能略有差异
                if et == "response.created":
                    print("• 会话已创建")
                elif et == "response.tool_call.created":
                    print("• 正在联网搜索…")
                elif et == "response.tool_call.delta":
                    # 某些版本会给出逐步的工具调用增量（可忽略或做更细日志）
                    pass
                elif et == "response.tool_call.completed":
                    print("• 搜索完成，正在整合证据…")
                elif et == "response.output_text.delta":
                    # 逐字输出正文
                    sys.stdout.write(event.delta)
                    sys.stdout.flush()
                elif et == "response.completed":
                    print("\n• 完成")
            final = stream_obj.get_final_response()
            # 可选：输出“推理摘要”
            try:
                rs = getattr(final, "reasoning", None)
                if rs and getattr(rs, "summary", None):
                    print("\n--- Reasoning Summary ---\n" + rs.summary.strip())
            except Exception:
                pass

    except OpenAIError as e:
        # 常见：账户暂未开通 web_search；给出友好提示并回退到非联网回答
        msg = str(e)
        print(f"\n[Warn] 流式失败：{msg}\n改用非流式/可能不联网的方式重试…\n")
        resp = client.responses.create(**payload)
        print(resp.output_text.strip())


def main():
    parser = argparse.ArgumentParser(
        description="Responses API + Web Search：边检索边推理的本地脚本"
    )
    parser.add_argument("query", type=str, help="你的问题/调研任务")
    parser.add_argument("--model", type=str, default="gpt-5",
                        help="模型（如 gpt-5 / gpt-5-thinking / o4-mini 等）")
    parser.add_argument("--stream", action="store_true",
                        help="流式输出进度与逐字文本")
    parser.add_argument("--sites", nargs="*", default=None,
                        help="仅在这些站点检索（空格分隔，如：arxiv.org aps.org berkeley.edu）")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: 请先在环境变量中设置 OPENAI_API_KEY")
        sys.exit(1)

    client = OpenAI()  # 会自动从 OPENAI_API_KEY 读取密钥
    call_responses_once(
        client=client,
        query=args.query,
        model=args.model,
        sites=args.sites,
        stream=args.stream,
    )


if __name__ == "__main__":
    main()