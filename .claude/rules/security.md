# Security Rules — Katabatic

- NEVER delete files in `raw_data/`, `discretized_data/`, or `Results/` without explicit user confirmation
- Never commit secrets or API keys — use environment variables
- Models under `katabatic/models_luke/` are Luke's working copies — do NOT auto-mirror to `katabatic/models/` without being asked
- Do not modify `katabatic/evaluate/tstr/evaluation.py` beyond seeded classifier fixes without user approval
