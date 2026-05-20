# Contributing

Thanks for taking interest in DocSearch. This project values clarity, offline reliability, and practical IR engineering.

## Ground rules
- Keep changes minimal and explain tradeoffs in the PR description.
- Avoid introducing cloud dependencies without a clear offline fallback.
- Prefer small, testable commits.

## Setup
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Project conventions
- Configuration lives in [src/config.py](src/config.py).
- Data artifacts are stored under [data/](data/) and are not committed.
- New scripts should go to tools/ with a short module docstring.

## Testing
There is no formal test suite yet. When touching retrieval or indexing, please:
- run a small indexing job
- run `python -m src.main --evaluate`
- include notes about the evaluation deltas in the PR

## Submitting a pull request
- Keep PRs focused (one feature/fix per PR).
- Include a short summary, affected areas, and any known limitations.
