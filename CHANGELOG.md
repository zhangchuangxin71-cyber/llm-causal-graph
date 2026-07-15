# Changelog

## v0.3.0 — 2026-07-15

Eval / recommendation stage fixes on top of the memory-safe training path.

### Changes
- **Use BPR-trained LightGCN at test time**: previous flow discarded the trained rec model and re-initialized embeddings for evaluation; ranking now uses the BPR-trained `rec_model`.
- **Configurable BPR stage**: add `--rec_epochs` (default 1) and `--rec_lr` (default 0.001).
- Train loop returns `rec_model` for evaluation.

### Files
- `train.py`
- `parameters.py`

---

## v0.2.0 — 2026-07-14

Adapt CausalDiffRec to train Yelp-scale graphs on a single 24GB GPU.

### Changes
- Graph editor: float16 `B`, edge-relative edit noise (keeps default `0.8` semantics vs sparse edges).
- Two-pass EERM training to avoid retaining multiple env graphs in memory.
- Chunked / scalar-weight VGAE loss; disable `detect_anomaly` for memory.
- Hyperparameters: `--env_k`, `--env_num`, `--edit_noise`.
- DNN concat dims fixed to `x + time_emb + env`.
- Loss `adjust_loss` argument order corrected.
- `.gitignore` for Python caches.

### Files
- `train.py`, `parameters.py`, `modules/generator.py`, `modules/DNN.py`, `utils/evaulate.py`, `utils/util_loss.py`, `.gitignore`

### Reproduce notes
- Best local Yelp@20 so far under official-like defaults (`lr=0.1`, `emd=8`, `steps=100`): Recall≈0.00794, NDCG≈0.00379 (paper Table II: 0.0120 / 0.0055).
- Prefer **one GPU** for full-graph runs to avoid CUDA OOM / device busy errors.
