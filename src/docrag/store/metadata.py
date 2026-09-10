from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_index(index_dir: str | Path, vectors: np.ndarray, chunks: list[dict]) -> None:
    """向量存 .npy,片段元数据存 .json。索引是"可重建的副产品"。"""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "vectors.npy", np.asarray(vectors, dtype="float32"))
    with open(index_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def load_index(index_dir: str | Path) -> tuple[np.ndarray, list[dict]]:
    index_dir = Path(index_dir)
    if not (index_dir / "vectors.npy").exists():
        raise FileNotFoundError(
            f"未找到索引({index_dir / 'vectors.npy'}),请先运行 `docrag index` 构建索引。"
        )
    vectors = np.load(index_dir / "vectors.npy")
    with open(index_dir / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return vectors, chunks
