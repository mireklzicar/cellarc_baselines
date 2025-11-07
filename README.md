# cellarc_baselines

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
