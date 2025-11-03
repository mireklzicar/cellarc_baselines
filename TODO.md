# TODO

- [x] Hook recursive reasoning configs into the dedicated ACT training pipeline, ensuring the schedule uses `lr=1e-4`, `lr_min_ratio=1.0`, and `lr_warmup_steps=2000`.
- [x] Restore puzzle-embedding training support for recursive reasoning models; current setup assumes flattened in-context episodes only.
- [x] Split `scripts/train.py` into `train_incontext.py` (full-episode flattening) and `train_embedding.py` (single I/O with puzzle embeddings) while preserving the existing CLI behaviour via thin wrappers or dispatch logic.
- [x] Extend the recursive embedding trainer with the remaining ACT niceties—cosine warm restarts, EMA, and puzzle-embedding optimizer splitting. Mirror the flow in `baselines/neural/recursive_reasoning/original_training/pretrain.py` (see `create_model`/`train_batch`) inside `scripts/train_embedding.py` and the helper utilities under `scripts/training/`.
- [x] Add guardrails so non-recursive baselines either train correctly under embedding mode or raise a clear error. The logic should live near the architecture checks in `scripts/train_embedding.py:191` and can reuse the registry in `baselines/__init__.py`.
- [x] Augment embedding-mode evaluation to measure held-out query performance (not just support reconstruction). Consider adding a query-forward path in `scripts/training/puzzle_embedding_dataset.py` that yields evaluation batches with `episode["query"]`/`episode["solution"]`, and update the evaluation loop in `scripts/train_embedding.py` accordingly.
