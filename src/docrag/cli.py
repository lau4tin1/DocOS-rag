from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import ask, build_index


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="docrag",
        description="从零构建的技术文档 RAG 检索助手",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("index", help="读取 data/raw 下的 Markdown,构建向量索引")

    p_ask = sub.add_parser("ask", help="根据索引回答一个问题")
    p_ask.add_argument("question", nargs="+", help="要问的问题(可用引号包裹)")
    p_ask.add_argument(
        "--show-sources",
        action="store_true",
        help="打印命中的片段来源与内容预览",
    )

    args = parser.parse_args()
    cfg = load_config()

    if args.command == "index":
        n = build_index(cfg)
        print(f"\n✓ 索引构建完成,共 {n} 个片段。")

    elif args.command == "ask":
        question = " ".join(args.question)
        answer, sources = ask(question, cfg)

        print("\n" + "=" * 40 + " 回答 " + "=" * 40)
        print(answer)

        if args.show_sources:
            print("\n" + "=" * 40 + " 来源 " + "=" * 40)
            for score, c in sources:
                print(f"\n[得分 {score:.4f}] {c.get('source')} — {c.get('section')}")
                print(c.get("text", "")[:200])


if __name__ == "__main__":
    main()
