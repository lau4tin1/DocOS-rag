from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
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


def _sse(event: dict) -> str:
    """把一个事件 dict 转成 Server-Sent Events 的一帧。"""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


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


@app.delete("/api/documents/{name}")
def delete_document(name: str):
    return engine.delete_file(name)


@app.post("/api/chat")
def chat(req: ChatRequest):
    cid = req.conversation_id or uuid.uuid4().hex
    history = conversations.get(cid, [])

    def event_stream():
        full = ""
        try:
            for ev in engine.stream_answer(req.message, history):
                if ev.get("type") == "delta":
                    full += ev.get("text", "")
                yield _sse(ev)
        finally:
            # 无论成功/失败,都把完整答案写入历史
            history.append({"role": "user", "content": req.message})
            history.append({"role": "assistant", "content": full})
            conversations[cid] = history[-(MAX_HISTORY_TURNS * 2):]

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# 前端静态文件:最后挂载,避免吞掉 /api 路由;html=True 使 "/" 直接返回 index.html
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


def main() -> None:
    import uvicorn

    uvicorn.run("docrag.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
