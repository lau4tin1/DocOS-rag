from docrag.engine import plan_index_update
from docrag.ingest.loader import hash_file

SIG = {"model": "M", "chunk_tokens": 256, "overlap_tokens": 32, "chunker": 3}


def _mk(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


def test_plan_unchanged(tmp_path):
    src = _mk(tmp_path, "a.md", "hello")
    manifest = {src: {"hash": hash_file(src), "config": SIG}}
    reuse, to_embed, deleted, _ = plan_index_update([{"source": src}], manifest, SIG)
    assert reuse[src] == src
    assert to_embed == set()
    assert deleted == set()


def test_plan_new_file(tmp_path):
    src = _mk(tmp_path, "a.md", "hello")
    reuse, to_embed, deleted, _ = plan_index_update([{"source": src}], {}, SIG)
    assert to_embed == {src}
    assert src not in reuse


def test_plan_rename_detected(tmp_path):
    old = _mk(tmp_path, "old.md", "same content")
    new = _mk(tmp_path, "new.md", "same content")  # 内容相同,路径不同
    manifest = {old: {"hash": hash_file(new), "config": SIG}}
    reuse, to_embed, deleted, _ = plan_index_update([{"source": new}], manifest, SIG)
    assert reuse[new] == old  # 改名:复用旧向量
    assert to_embed == set()  # 不重新向量化
    assert deleted == set()   # 改名不算删除


def test_plan_content_changed(tmp_path):
    src = _mk(tmp_path, "a.md", "v1")
    manifest = {src: {"hash": "different-hash", "config": SIG}}
    reuse, to_embed, deleted, _ = plan_index_update([{"source": src}], manifest, SIG)
    assert to_embed == {src}


def test_plan_deleted(tmp_path):
    src = _mk(tmp_path, "a.md", "hello")
    manifest = {
        src: {"hash": hash_file(src), "config": SIG},
        "gone.md": {"hash": "x", "config": SIG},
    }
    reuse, to_embed, deleted, _ = plan_index_update([{"source": src}], manifest, SIG)
    assert deleted == {"gone.md"}


def test_plan_empty_file_is_unchanged(tmp_path):
    # 空文件(0 片段)应判定为"未变化",而不是每次重跑
    src = _mk(tmp_path, "empty.txt", "")
    manifest = {src: {"hash": hash_file(src), "config": SIG}}
    reuse, to_embed, deleted, _ = plan_index_update([{"source": src}], manifest, SIG)
    assert reuse[src] == src
    assert to_embed == set()
