# cellarc_baselines

This repository contains model training and baselines for CellARC:
- Dataset repo: https://github.com/mireklzicar/cellarc 
- Website: https://cellarc.mireklzicar.com/

## Wandb Results

| Project | Size | Training Mode | W&B URL |
| --- | --- | --- | --- |
| `cellarc100k_50e_embedding_small` | small | embedding | https://wandb.ai/lzicar/cellarc100k_50e_embedding_small |
| `cellarc100k_50e_embedding_medium` | medium | embedding | https://wandb.ai/lzicar/cellarc100k_50e_embedding_medium |
| `cellarc100k_50e_embedding_large` | large | embedding | https://wandb.ai/lzicar/cellarc100k_50e_embedding_large |
| `cellarc100k_50e_incontext_small` | small | incontext | https://wandb.ai/lzicar/cellarc100k_50e_incontext_small |
| `cellarc100k_50e_incontext_medium` | medium | incontext | https://wandb.ai/lzicar/cellarc100k_50e_incontext_medium |
| `cellarc100k_50e_incontext_large` | large | incontext | https://wandb.ai/lzicar/cellarc100k_50e_incontext_large |

## A. Basic (Single-GPU) Training
- Use `python scripts/train.py --config-name train/default` with Hydra overrides for architecture/size/mode.
- Example (tiny_recursive large embedding with W&B logging):
  ```bash
  python scripts/train.py \
    --config-name train/default \
    model.architecture=tiny_recursive \
    model/size=large \
    training.mode=embedding \
    trainer.checkpoints.enabled=true \
    logging.wandb.enabled=true \
    logging.wandb.project=cellarc100k_50e_embedding_large \
    logging.wandb.group=mode_embedding \
    logging.wandb.name=tiny_recursive_large_embedding_single
  ```
- Smoke test: swap to the lightweight `train/smoke` config for a 5-step sanity check before long runs:
  ```bash
  python scripts/train.py \
    --config-name train/smoke \
    model.architecture=tiny_recursive \
    model/size=small \
    training.mode=incontext \
    trainer.checkpoints.enabled=false \
    logging.wandb.enabled=false
  ```

## B. Tmux Parallelism (Multi-Run Scheduling)
- `scripts/train_all_tmux.sh` splits independent runs across GPUs via tmux workers.
- Launch a curated subset on GPUs 0–3:
  ```bash
  bash scripts/train_all_tmux.sh --gpus 0,1,2,3 \
    --run transformer_act:large:embedding \
    --run tiny_recursive:large:embedding \
    --run hrm:large:embedding \
    --run transformer:large:incontext
  ```
- Status lives under `outputs/tmux_runs/<timestamp>`; attach to sessions with `tmux attach -t train_all_gpu0`.
- Smoke test: `bash scripts/train_all_smoke_test.sh small` runs every architecture in both modes with the `train/smoke` config (5 optimizer steps) before you queue the larger tmux batch; pass `small medium` to limit sizes.

## C. Torch Distributed (Multi-GPU Single Run)
- For data-parallel training of one embedding run across 4 GPUs use `torchrun` (rank 0 handles logging).
  ```bash
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:true \
  torchrun --nproc-per-node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    -m scripts.train --config-name train/default \
    model.architecture=tiny_recursive \
    model/size=large \
    training.mode=embedding \
    data.batch_size=96 \
    trainer.gradient_accumulation=2 \
    trainer.checkpoints.enabled=true \
    logging.wandb.enabled=true \
    logging.wandb.project=cellarc100k_50e_embedding_large \
    logging.wandb.group=embedding_ddp \
    logging.wandb.name=tiny_recursive_large_embedding_ddp
  ```
- Per-rank batch plus gradient accumulation controls memory footprint; set `wandb login` once before running.
- Smoke test: keep the same `torchrun` invocation but point at `train/smoke` so the job exits after a handful of steps:
  ```bash
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:true \
  torchrun --nproc-per-node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    -m scripts.train --config-name train/smoke \
    model.architecture=tiny_recursive \
    model/size=small \
    training.mode=embedding \
    logging.wandb.enabled=false \
    trainer.checkpoints.enabled=false
  ```

## D. Symbolic Baseline Evaluation
- Run all bundled symbolic solvers with a single command; results land in `outputs/symbolic/`:
  ```bash
  bash scripts/run_symbolic_baselines.sh
  ```
- To smoke-test or target one solver, call the Hydra entry point directly and bound the episode count:
  ```bash
  python scripts/eval_symbolic.py \
    baseline.name=copycat \
    eval.max_episodes=16 \
    eval.progress_bar=true
  ```

## E. LLM Evaluation
- Evaluate GPT-based baselines (defaults to `gpt-5-2025-08-07`) over the 100-episode HF splits; ensure your OpenAI credentials are exported before running:
  ```bash
  bash scripts/run_gpt_eval.sh
  ```
- Smoke test: limit to 10 episodes and write to a `_smoke` prediction log by toggling the environment flag:
  ```bash
  SMOKE_TEST=true bash scripts/run_gpt_eval.sh
  ```
- Batch multiple hosted LLMs via Hydra overrides (or reuse `scripts/run_llm_baselines.sh`) when you need to sweep model names.
