# Baselines

This directory collects a handful of baselines that can be run against the
processed ARC-style benchmarks in `data/benchmark_1M_processed`. The code is
organised into three subpackages:

- `neural/` – PyTorch sequence models wired into the unified training loop via
  `baselines.create_baseline`.
- `symbolic/` – Lightweight algorithmic solvers available through
  `baselines.symbolic.create_symbolic_baseline` and the Hydra-driven evaluation
  script `scripts/eval_symbolic.py`.
- `llm/` – Placeholders for large-language-model-based approaches.

## Included models

- **RNN (`rnn`)** – A PyTorch reimplementation of the recurrent models
  originally released with the Chomsky hierarchy benchmarks
  ([source](https://github.com/mireklzicar/neural_networks_chomsky_hierarchy_torch)).
  We wrap the LSTM variant in a thin seq2seq adapter.
- **Transformer (`transformer`)** – The encoder/decoder architecture from the
  same Chomsky hierarchy project, exported here with minimal changes and a
  helper factory.
- **1D CNN (`cnn1d`)** – A lightweight convolutional baseline we added that
  embeds tokens, applies two Conv1d+GELU blocks, and emits per-token logits.
- **Tiny Recursive Model (`tiny_recursive`)** – The Tiny Recursive Model (TRM)
  from Samsung SAIL Montreal
  ([source](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)),
  configured to take a single ACT step.
- **Hierarchical Reasoning Model (`hrm`)** – The HRM baseline from Sapient AI
  ([source](https://github.com/sapientinc/HRM)). At the moment we only expose
  the model code; wiring it into the unified training loop would require
  additional batching logic similar to the TRM wrapper.

All recursive reasoning utilities from the upstream projects are vendored under
`baselines/neural/recursive_reasoning`.
