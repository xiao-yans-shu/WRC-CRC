const API_BASE_URL = window.API_BASE_URL || "http://127.0.0.1:8000";

const elements = {
  fileInput: document.getElementById("file-input"),
  fileHint: document.getElementById("file-hint"),
  codeInput: document.getElementById("code-input"),
  submitBtn: document.getElementById("submit-btn"),
  resultLabel: document.getElementById("result-label"),
  resultProb: document.getElementById("result-prob"),
  resultFilename: document.getElementById("result-filename"),
  messages: document.getElementById("messages"),
};

let currentFilename = "";

const updateMessage = (text, type = "info") => {
  elements.messages.textContent = text;
  elements.messages.dataset.type = type;
};

const resetMessages = () => {
  updateMessage("");
};

const setLoading = (status) => {
  elements.submitBtn.disabled = status;
  elements.submitBtn.textContent = status ? "正在推理..." : "开始分类";
};

const renderResult = (label, probability, filename) => {
  elements.resultLabel.classList.remove("positive", "negative", "neutral");
  const chineseLabel = label === "Readable" ? "可读" : "不可读";
  const cssClass = label === "Readable" ? "positive" : "negative";
  elements.resultLabel.classList.add(cssClass);
  elements.resultLabel.textContent = chineseLabel;
  elements.resultProb.textContent = `模型置信度：${(probability * 100).toFixed(
    2
  )}%`;
  elements.resultFilename.textContent = filename
    ? `来自：${filename}`
    : "来自：-";
};

const callInferenceApi = async (payload) => {
  const response = await fetch(`${API_BASE_URL}/classify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code: payload }),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "推理失败");
  }
  return data;
};

const handleSubmit = async () => {
  resetMessages();
  const code = elements.codeInput.value;

  if (!code.trim()) {
    updateMessage("请先粘贴代码或上传文件。", "warning");
    return;
  }

  setLoading(true);
  elements.resultLabel.classList.add("neutral");
  elements.resultLabel.textContent = "分析中...";
  elements.resultProb.textContent = "";
  try {
    updateMessage("正在调用模型...", "info");
    const result = await callInferenceApi(code);
    renderResult(result.label, result.probability, currentFilename);
    updateMessage("推理成功。", "success");
  } catch (error) {
    console.error(error);
    updateMessage(error.message || "推理失败，请检查日志。", "error");
    elements.resultLabel.classList.add("neutral");
    elements.resultLabel.textContent = "发生错误";
    elements.resultProb.textContent = "";
  } finally {
    setLoading(false);
  }
};

const handleFileSelection = async (event) => {
  const file = event.target.files[0];
  if (!file) {
    elements.fileHint.textContent = "尚未选择文件";
    currentFilename = "";
    return;
  }

  currentFilename = file.name;
  elements.fileHint.textContent = `已加载：${file.name} (${(
    file.size /
    1024
  ).toFixed(1)} KB)`;

  try {
    const text = await file.text();
    elements.codeInput.value = text;
    updateMessage("文件内容已载入文本框，可直接修改后提交。", "info");
  } catch (error) {
    console.error(error);
    updateMessage("读取文件失败，请确认编码格式。", "error");
  }
};

elements.fileInput.addEventListener("change", handleFileSelection);
elements.submitBtn.addEventListener("click", handleSubmit);

// 初始化状态
renderResult("Readable", 0.0, "");
elements.resultLabel.classList.add("neutral");
elements.resultLabel.textContent = "等待提交";

