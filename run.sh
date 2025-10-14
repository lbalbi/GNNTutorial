#!/bin/bash
#
#SBATCH --job-name=check_cuda
#SBATCH --output=check_cuda_%j.log
#SBATCH --error=check_cuda_%j.err
#SBATCH --ntasks-per-node=1         # one task (shell) per node
#SBATCH --time=00:05:00
#SBATCH --partition=tier1

# # 1
python setup.py
# #2

uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), flush=True)"

# #3
uv run python main.py