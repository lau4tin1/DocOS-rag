from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import ask, build_index, delete_file, list_documents


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="docrag",
        description="从零构建的技术文档 RAG 检索助手",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("index", help="读取 data/raw 下的文档,构建向量索引")
    sub.add_parser("list", help="列出已索引的文件")

    p_ask = sub.add_parser("ask", help="根据索引回答一个问题")
    p_ask.add_argument("question", nargs="+", help="要问的问题(可用引号包裹)")
    p_ask.add_argument(
        "--show-sources",
        action="store_true",
        help="打印命中的片段来源与内容预览",
    )

    p_del = sub.add_parser("delete", help="删除一个文件及其 chunk/向量")
    p_del.add_argument("filename", help="文件名(可用 docrag list 查看)")

    args = parser.parse_args()
    cfg = load_config()

    if args.command == "index":
        n = build_index(cfg)
        print(f"\n✓ 索引构建完成,共 {n} 个片段。")

    elif args.command == "list":
        docs = list_documents(cfg)
        if not docs:
            print("还没有索引任何文档。")
        else:
            print(f"已索引 {len(docs)} 个文件:")
            for d in docs:
                print(f"  {d['name']}  ({d['chunks']} 片段)")

    elif args.command == "delete":
        result = delete_file(args.filename, cfg)
        if result["found"]:
            print(f"\n✓ 已删除 {result['deleted']},当前共 {result['total_chunks']} 个片段。")
        else:
            print(f"\n未找到文件 {result['deleted']},索引未变(用 docrag list 查看文件名)。")

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
