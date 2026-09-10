#!/usr/bin/env python3
"""不安装、直接运行的入口:python cli.py index / ask。"""
from __future__ import annotations

import sys
from pathlib import Path

# 把 src 加入导入路径,这样无需 pip install 也能运行
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from docrag.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
