# iTool: Reinforced Fine-Tuning with Dynamic Deficiency Calibration

This repository contains the code for the paper "iTool: Reinforced Fine-Tuning with Dynamic Deficiency Calibration for Advanced Tool Use", built on top of LLaMA-Factory.

Summary: We extend LLaMA-Factory with customizations for Agent / MCTS training and DPO training. Most modifications live under the `mcts/` directory. The repository includes a script to run the iTool experiments: `mcts/run_mcts_dpo.sh`.

- ## Structure & Key Files
- `mcts/` — Core code for this paper: MCTS, DPO training logic and helper scripts.
- `mcts/run_mcts_dpo.sh` — Main script to start experiments (intended for Linux / WSL / Git Bash).
- `src/` — The LLaMA-Factory core package (`llamafactory`).
- `scripts/cal_ppl.py` — Compute model perplexity (PPL) on a dataset.

## Quick Start (Developer)
1. Clone and install dependencies (recommended to use a virtualenv or conda):

```powershell
# Windows (PowerShell)
git clone <repo-url>
cd iTool
pip install -e ."[torch,metrics]"
```

2. If you are on Windows and need to run shell scripts, use WSL or Git Bash:

```powershell
# From Git Bash or WSL, run:
bash mcts/run_mcts_dpo.sh
```

3. The script will start the training/evaluation workflow using parameters defined under `mcts`. For step-by-step debugging, open the script to inspect the invoked Python commands and environment variables, then run the corresponding Python modules directly (for example: `python -m mcts.main`).

## Notes
- The project follows LLaMA-Factory conventions: configuration and run-time parameters are primarily provided via YAML files in `examples/`.
- For GPU / multi-GPU / Deepspeed / quantization setups, see `requirements.txt`, `setup.py` extras, and the main repository README for CUDA / NPU / Docker guidance.
- On Windows, when using QLoRA / bitsandbytes, ensure you have a compatible `bitsandbytes` wheel installed.

## Citation
If this work is useful to you, please cite the paper. Recommended BibTeX entry:

```bibtex
@article{zeng2025itool,
  title={iTool: Reinforced Fine-Tuning with Dynamic Deficiency Calibration for Advanced Tool Use},
  author={Zeng, Yirong and Ding, Xiao and Wang, Yuxian and Liu, Weiwen and Ning, Wu and Hou, Yutai and Huang, Xu and Qin, Bing and Liu, Ting},
  journal={arXiv preprint arXiv:2501.09766},
  year={2025}
}
```

Acknowledgement: This project builds on LLaMA-Factory — please acknowledge it when appropriate.
