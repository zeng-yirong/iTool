## 快速目标

这是为 AI 编码代理准备的简明指南，帮助你快速理解并修改 iTool（基于 LLaMA-Factory）代码库的关键点、常用开发流程和可编辑位置。

## 项目大局（必读）
- 包布局：源码在 `src/` 下，发布包名为 `llamafactory`（见 `setup.py` 的 `package_dir`）。
- 核心入口：
  - CLI：`llamafactory-cli`（在 `setup.py` 中通过 entry_points 注册，兼容短命令 `lmf`）。
  - API 服务：`src/api.py` — 启动 FastAPI/uvicorn，使用 `llamafactory.api.app.create_app`。
  - 训练启动：`src/train.py` 调用 `llamafactory.train.tuner.run_exp`。
  - 浏览器 UI：`src/webui.py` 使用 Gradio，`llamafactory-cli webui` 可启动。

## 主要开发/运行命令（可直接执行）
- 安装（可编辑依赖或启用 extras）：
  ```bash
  pip install -e ."[torch,metrics]"
  # 或 (Windows) pip install -e ."[torch,metrics]"
  ```
- 快速示例：
  - 微调：`llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml`
  - 推理：`llamafactory-cli chat examples/inference/llama3_lora_sft.yaml`
  - 合并 LoRA：`llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml`
  - API（vLLM 示例）：`API_PORT=8000 llamafactory-cli api examples/inference/llama3_vllm.yaml`
  - Web UI：`llamafactory-cli webui`

## 配置与约定（代码修改时务必遵守）
- 实验由 YAML 驱动，所有训练/推理变更优先在 `examples/*.yaml` 中体现。修改训练流程请优先修改/新增 YAML 并在 `train.tuner` 中保持兼容。示例位置：`examples/train_lora`。
- 模型/模板约定：对话模型必须使用对应模板（见 `src/llamafactory/data/template.py`），模型列表在 `src/llamafactory/extras/constants.py`。
- 包风格：项目使用 `ruff` 配置（见 `pyproject.toml`），line-length=119，双引号为首选引号。

## 集成点 & 外部依赖（修改或新增功能时检查）
- 依赖管理：`requirements.txt` + `setup.py` 的 `extras_require`（如 `vllm`, `bitsandbytes`, `deepspeed`, `torch-npu` 等）。
- 存储/模型源：项目支持 Hugging Face 与 ModelScope（环境变量 `USE_MODELSCOPE_HUB=1`）。
- 容器化：`docker/docker-*/docker-compose.yml` 提供 CUDA/NPU/ROCm 镜像和常见启动步骤，修改 Dockerfile 时请同步 `docker/` 下对应目录。

## 代码变更建议（AI agent 可直接执行的策略）
- 若新增 CLI 行为：在 `src/llamafactory/cli.py`（或相应目录）添加子命令并更新 `setup.py` entry_points（注意 `ENABLE_SHORT_CONSOLE` 环境变量）。
- 新增训练/评估模块：遵循 `train.tuner.run_exp` 的调用约定，优先增加 YAML 配置并复用现有 utils 与 dataset 加载逻辑（见 `data/` 和 `src/llamafactory/train/`）。
- 小改动（样式/检查）：使用 `ruff`（配置见 `pyproject.toml`）和 `pytest`（`extras_require.dev`）运行快速验证。

## 调试与验证要点
- 本地快速验证：编辑小规模 YAML（batch 小、max_steps 小）并运行 `llamafactory-cli train <yaml>`。这可触发配置解析、数据加载与主训练路径。
- API 验证：运行 `src/api.py`，打开 `http://localhost:8000/docs` 查看自动生成的 API 文档。

## 重要文件索引（快速跳转）
- `README.md`, `README_zh.md` — 高阶说明与示例命令。
- `setup.py` — 包名、entry points 和 extras。
- `pyproject.toml` — ruff 配置。
- `requirements.txt` — 基础依赖版本约束。
- `examples/` — YAML 实验样例（首选改动位置）。
- `src/llamafactory/` — 包的核心实现（model/chat/train/webui/api 等子模块）。

## 如果不确定，优先顺序
1. 在 `examples/*.yaml` 中添加/修改配置并复现。2. 在 `src/llamafactory/*` 中实现必要的接口更改（保持向后兼容）。3. 更新 `requirements.txt` / `setup.py` extras（需小心破坏现有环境）。

---
请检查这份草稿并告诉我：想保留 README 中哪些已有段落（例如中文安装、NPU 指南或 Docker 细节），或希望我把某些内部路径（如 `src/llamafactory/train` 的更细粒度文件）加入到指令中。
