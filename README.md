# Implementation of Multi-Game Decision Transformers in PyTorch

## Quickstart
```bash
conda create --name mgdt python=3.10
conda activate mgdt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

python scripts/download_weights.py
python run.py
```

## Baselines

> [logs](workdir/)

| model | params | task     | this repo | orig. |
| ----- | ------ | -------- | --------- | ----- |
| mgdt  | 200M   | Breakout | 298.8     | 290.6 |

## References:

- [1] Original code in Jax: https://github.com/google-research/google-research/tree/master/multi_game_dt
- [2] Paper: https://arxiv.org/abs/2205.15241
