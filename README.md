### Title: 
A Weakly-Supervised Contrastive Learning Framework for Few-Shot Code Readability Classification (WCR-CLC)

### Introduction: 
This project provide the code of WCR-CLC.

### Installation:  
python 3.10

Bert

### Usage:  
Please read our article first to understand the code process as well as the datasets corresponding to pre-training and fine-tuning.

For the token-based backbone network, we first run the token-based_pre.py for pre-training and save the generated model weights in the s.h5. Then, we run fine-texture.py for fine-tuning to obtain the results. During this period, please make sure to use the correct dataset.

For the character-based backbone network, the overall process is consistent with the above, with model weights saved in the t.h5. However, it is necessary to change the preprocess_structure_data() function and adjust the representation dimensions in the triplet_model_stru.py and fine_stru.py according to the dataset used during training.

## 推理前端与API

为了在无 TensorFlow 环境的机器上编辑代码、再到有环境的机器上运行，我们新增了一个独立的推理服务与前端页面：

### 目录说明

- `webapp/backend/`：FastAPI 推理服务，复现 `fine_stru.py` 的结构模型并加载 `T_BEST.hdf5` 权重。
- `webapp/frontend/`：纯静态页面，支持上传或粘贴代码，调用后端接口获取预测结果。

### 部署步骤（在已安装 TensorFlow 的机器上执行）

1. 进入仓库根目录，创建虚拟环境（可选）并安装依赖：
   ```
   pip install -r webapp/backend/requirements.txt
   ```
2. 确保模型权重存在，默认路径为 `code/Experimental output/T_BEST.hdf5`。如路径不同，可在启动前设置环境变量：
   ```
   set MODEL_WEIGHTS_PATH=绝对路径\到\你的权重.hdf5
   ```
3. 启动接口：
   ```
   uvicorn webapp.backend.app:app --host 0.0.0.0 --port 8000
   ```
4. 打开/部署前端：
   - 直接双击 `webapp/frontend/index.html`（使用本地浏览器调用 `http://127.0.0.1:8000`）。
   - 如需远程访问，可用任意静态服务器托管该目录（例如 `npx serve webapp/frontend`），并在 `webapp/frontend/app.js` 中修改 `API_BASE_URL`。

前端支持上传文件或直接粘贴代码，内部会将文本编码成 50×305 的矩阵，与训练阶段保持一致，并显示“可读/不可读”标签及概率。

### Python 3.6 兼容方案（Flask）

如无法升级到 Python ≥3.8，可使用备用的 Flask 服务：

1. 安装专用依赖（兼容 3.6）：
   ```
   pip install -r webapp/backend/requirements-py36.txt
   ```
2. 启动 Flask 版本接口：
   ```
   python -m webapp.backend.flask_server
   ```
   若权重路径不同，依旧可通过 `MODEL_WEIGHTS_PATH` 环境变量覆盖。
3. 前端调用方式与 FastAPI 版本一致，无需改动页面。

注意：`requirements-py36.txt` 使用 TensorFlow 2.4.4 与适配的 numpy 版本，请确保显卡驱动、CUDA 对应该版本的兼容矩阵。