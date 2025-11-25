import logging
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request
from flask_cors import CORS

from .classifier import StructureReadabilityClassifier, default_weights_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("readability_flask")

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=True,
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.after_request
def apply_cors(response):
    response.headers.setdefault("Access-Control-Allow-Origin", "*")
    response.headers.setdefault("Access-Control-Allow-Headers", "*")
    response.headers.setdefault("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

_classifier: Optional[StructureReadabilityClassifier] = None


def get_classifier() -> StructureReadabilityClassifier:
    global _classifier
    if _classifier is None:
        weights_path: Path = default_weights_path()
        logger.info("加载模型权重: %s", weights_path)
        _classifier = StructureReadabilityClassifier(weights_path)
    return _classifier


@app.before_first_request
def load_model():
    get_classifier()
    logger.info("模型加载完成（Flask）")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/classify", methods=["POST"])
def classify():
    payload = request.get_json(force=True, silent=True) or {}
    code = (payload.get("code") or "").strip("\n")
    if not code.strip():
        return jsonify({"detail": "代码内容不能为空"}), 400

    result = get_classifier().predict_label(code)
    return jsonify(result)


@app.route("/classify-file", methods=["POST"])
def classify_file():
    if "file" not in request.files:
        return jsonify({"detail": "未检测到文件"}), 400
    uploaded_file = request.files["file"]
    raw_bytes = uploaded_file.read()
    if not raw_bytes:
        return jsonify({"detail": "文件为空"}), 400

    text = None
    for encoding in ("utf-8", "gbk", "latin1"):
        try:
            text = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        return jsonify({"detail": "无法解析文件编码"}), 400
    if not text.strip():
        return jsonify({"detail": "文件内容为空"}), 400

    result = get_classifier().predict_label(text)
    result["filename"] = uploaded_file.filename
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)

