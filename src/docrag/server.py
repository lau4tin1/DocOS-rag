from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import load_config
from .engine import RAGEngine

ALLOWED_EXTS = {".md", ".markdown", ".txt", ".pdf"}
MAX_HISTORY_TURNS = 20  # 每个会话最多保留的往返轮数

cfg = load_config()
engine = RAGEngine(cfg)
engine.reload()  # 启动时加载已有索引

# 会话历史(内存,重启清空):conversation_id -> [{role, content}, ...]
conversations: dict[str, list[dict]] = {}

app = FastAPI(title="DocRAG 检索助手")

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    uploaded: list[str] = []
    skipped: list[str] = []
    for f in files:
        name = Path(f.filename or "").name  # 去掉可能的目录路径,防路径穿越
        ext = Path(name).suffix.lower()
        if not name or ext not in ALLOWED_EXTS:
            skipped.append(f.filename or "(空)")
            continue
        (Path(cfg.paths.raw_dir) / name).write_bytes(await f.read())
        uploaded.append(name)

    total = engine.build_index() if uploaded else len(engine.chunks)
    return {"uploaded": uploaded, "skipped": skipped, "total_chunks": total}


@app.get("/api/documents")
def documents():
    return engine.list_documents()


@app.post("/api/chat")
def chat(req: ChatRequest):
    cid = req.conversation_id or uuid.uuid4().hex
    history = conversations.get(cid, [])

    try:
        answer, results = engine.answer(req.message, history)
        sources = [
            {
                "score": round(score, 4),
                "source": chunk.get("source", ""),
                "section": chunk.get("section", ""),
                "text": chunk.get("text", "")[:300],
            }
            for score, chunk in results
        ]
    except Exception as e:  # 例如没配 API key、网络失败等,返回友好信息而非 500
        answer = f"生成失败:{e}"
        sources = []

    # 更新历史(限制长度,避免无限增长)
    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": answer})
    conversations[cid] = history[-(MAX_HISTORY_TURNS * 2):]

    return {"conversation_id": cid, "answer": answer, "sources": sources}


# 前端静态文件:最后挂载,避免吞掉 /api 路由;html=True 使 "/" 直接返回 index.html
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


def main() -> None:
    import uvicorn

    uvicorn.run("docrag.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
