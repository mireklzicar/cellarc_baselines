# TODO

- [ ] Hook recursive reasoning configs into the dedicated ACT training pipeline, ensuring the schedule uses `lr=1e-4`, `lr_min_ratio=1.0`, and `lr_warmup_steps=2000`.
- [ ] Restore puzzle-embedding training support for recursive reasoning models; current setup assumes flattened in-context episodes only.
- [ ] Split `scripts/train.py` into `train_incontext.py` (full-episode flattening) and `train_embedding.py` (single I/O with puzzle embeddings) while preserving the existing CLI behaviour via thin wrappers or dispatch logic.
- [ ] Implement a dedicated recursive-reasoning training loop (mirroring the structure in `baselines/neural/recursive_reasoning/original_training/pretrain.py`) that wires in ACT-specific components such as `AdamATan2`, EMA support, and the loss scheduling we rely on.
- [ ] Ensure both training modes (in-context vs. embedding) can target both the generic seq2seq baselines and the recursive reasoning models, selecting the correct pipeline automatically based on the `model.architecture` choice.
