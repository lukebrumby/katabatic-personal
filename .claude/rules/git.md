# Git Rules — Katabatic

- NEVER commit directly to main — always use feature branches
- Never use `git add -A` or `git add .` — add files individually by name
- Never amend published commits — create new ones
- Never skip pre-commit hooks (--no-verify)
- Never force push to main/master
- Run `poetry run pytest` before marking any task complete
- Never commit without being explicitly asked — show the diff and wait
