# Repository Guidelines

## Project Structure & Module Organization
- `scripts/`: Python CLI tools (e.g., `srx-to-project.py`, `cluster-ontology.py`) and helpers (e.g., `db_utils.py`). New CLIs live here; prefer argparse with `if __name__ == "__main__":`.
- `notebooks/`: Analysis notebooks grouped by topic (e.g., `CellXGene/`, `SRAgent/`). Clear outputs before committing; commit small tables/figures only when essential.
- `nextflow/metaq/`: Nextflow pipeline (`main.nf`, `nextflow.config`) for H5AD processing; requires `run_metaq` in your `PATH` and an HPC profile (e.g., SLURM).
- `data/`: Small reference/lookup data used by notebooks/scripts. Do not commit large binaries.

## Build, Test, and Development Commands
- Python env: `python -m venv .venv && source .venv/bin/activate`
- Run a script: `python scripts/srx-to-entrez-id.py --help`
- Run notebooks: `jupyter lab` (keep outputs minimal before commit).
- Run Nextflow: `nextflow run nextflow/metaq/main.nf -resume -profile slurm --input_dir <in> --output_dir <out>`

## Coding Style & Naming Conventions
- Python: PEP 8, 4‑space indentation, snake_case for functions/variables. Typing where practical; add module/function docstrings.
- Filenames: CLIs may use kebab-case; importable helpers use snake_case (e.g., `sql_db_utils.py`).
- Prefer small, composable functions to enable testing. Keep user‑visible CLIs stable.

## Testing Guidelines
- Framework: pytest. Place tests in `tests/` as `test_<module>.py`.
- Target pure logic (e.g., parsers, mappers); mock network/IO. Aim for smoke coverage of key CLIs via `--help` and small fixtures in `data/`.
- Run: `pytest -q` (add as available).

## Commit & Pull Request Guidelines
- Commits: concise subject (≤72 chars), imperative mood; body explains what/why and notable data changes. Example: `notebooks: add UMI summary and CSV export`.
- PRs: focused scope, description, paths touched, sample command or screenshot for notebook/plot changes, and linked issue if applicable. Strip notebook outputs or keep to informative cells only.

## Security & Configuration Tips
- Never commit credentials. Use env vars (e.g., `LANGSMITH_API_KEY` for `langsmith-run-summary.py`; GCP/DB creds managed outside the repo).
- For HPC runs, set correct Nextflow profile/queue in `nextflow/metaq/nextflow.config` and ensure `run_metaq` is installed.

## Agent-Specific Instructions
- Keep edits surgical; preserve existing structure and filenames unless discussed.
- New utilities go in `scripts/`; prefer argparse, `--help`, and sane defaults. Update `README.md` if behavior or paths change.
