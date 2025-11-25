import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .classifier import (
    StructureReadabilityClassifier,
    default_weights_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("readability_api")


class CodeSnippet(BaseModel):
    code: str


def _init_classifier(weights_path: Optional[Path] = None) -> StructureReadabilityClassifier:
    resolved_path = weights_path or default_weights_path()
    logger.info("加载模型权重: %s", resolved_path)
    return StructureReadabilityClassifier(resolved_path)


classifier_holder: dict[str, StructureReadabilityClassifier] = {}


def get_classifier() -> StructureReadabilityClassifier:
    if "instance" not in classifier_holder:
        classifier_holder["instance"] = _init_classifier()
    return classifier_holder["instance"]


app = FastAPI(
    title="代码可读性推理服务",
    version="1.0.0",
    description="基于WCR-CLC结构分支的代码可读性分类API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    try:
        get_classifier()
        logger.info("模型加载完成")
    except Exception as exc:
        logger.exception("模型加载失败: %s", exc)
        raise


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/classify")
async def classify_snippet(payload: CodeSnippet):
    code = (payload.code or "").strip("\n")
    if not code.strip():
        raise HTTPException(status_code=400, detail="代码内容不能为空")

    result = get_classifier().predict_label(code)
    return result


@app.post("/classify-file")
async def classify_file(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="文件为空")

    text = None
    for encoding in ("utf-8", "gbk", "latin1"):
        try:
            text = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        raise HTTPException(status_code=400, detail="无法解析文件编码")

    if not text.strip():
        raise HTTPException(status_code=400, detail="文件内容为空")

    result = get_classifier().predict_label(text)
    result["filename"] = file.filename
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("webapp.backend.app:app", host="0.0.0.0", port=8000, reload=True)

