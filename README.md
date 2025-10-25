# iTool: Reinforced Fine-Tuning with Dynamic Deficiency Calibration

这是论文 “iTool: Reinforced Fine-Tuning with Dynamic Deficiency Calibration for Advanced Tool Use” 的代码实现，基于 LLaMA-Factory 开发。

简要说明：本仓库在 LLaMA-Factory 的基础上做了针对 Agent/MCTS 训练与 DPO 训练的定制修改，主要集中在 `mcts/` 目录下。该实现包含用于启动 iTool 实验的脚本 `mcts/run_mcts_dpo.sh`，可用于复现论文中描述的训练流程。

## 目录与关键文件
- `mcts/` — 包含 MCTS、DPO 训练逻辑与相关实用脚本。
- `mcts/run_mcts_dpo.sh` — 启动实验的主脚本（在 Linux / WSL / Git Bash 下运行）。
- `src/` — 来自 LLaMA-Factory 的核心包（`llamafactory`）。
- `scripts/cal_ppl.py` — 计算模型在数据集上的 perplexity（PPL）。

## 快速开始（开发者环境）
1. 克隆并安装依赖（推荐在虚拟环境或 conda 中）：

```powershell
# Windows (PowerShell)
git clone <repo-url>
cd iTool
pip install -e ."[torch,metrics]"
```

2. 如果你在 Windows 上运行 shell 脚本，建议使用 WSL 或 Git Bash：

```powershell
# 使用 Git Bash 或 WSL 进入仓库根目录，然后：
bash mcts/run_mcts_dpo.sh
```

3. 脚本会根据 `mcts` 内的参数启动训练/评估流程。若需要在本地逐步调试，可打开脚本并查看传入的 Python 命令与环境变量，然后直接运行对应的 Python 模块（例如通过 `python -m mcts.main` 形式运行）。

## 运行注意事项
- 本项目继承了 LLaMA-Factory 的依赖和习惯：训练/推理的参数优先通过 YAML（`examples/`）指定。
- 如果使用 GPU/多卡/Deepspeed/量化等功能，请参考 `requirements.txt`、`setup.py` 的 `extras_require` 与仓库根 README 中的 CUDA/NPU/Docker 指南。
- 在 Windows 环境上做 QLoRA/bitsandbytes 时，请确保安装兼容的 `bitsandbytes` wheel（如 README 中所述）。

## 引用
如果你对本工作感兴趣，请引用论文，推荐的 BibTeX 条目：

```bibtex
@article{zeng2025itool,
	title={iTool: Reinforced Fine-Tuning with Dynamic Deficiency Calibration for Advanced Tool Use},
	author={Zeng, Yirong and Ding, Xiao and Wang, Yuxian and Liu, Weiwen and Ning, Wu and Hou, Yutai and Huang, Xu and Qin, Bing and Liu, Ting},
  	journal={arXiv preprint arXiv:2501.09766},
  	year={2025}
}
```

并同时致谢 LLaMA-Factory（基础框架）。
