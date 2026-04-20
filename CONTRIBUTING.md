# Contributing to D-HAQNAS

Thanks for your interest in contributing.

## Ways to contribute

- Report bugs or request features via [GitHub Issues](https://github.com/sanvisharma850/Differentiable-Hardware-Aware-Quantum-Neural-Architecture-Search/issues).
- Improve documentation (README, figures, reproducibility notes).
- Improve or extend experimental workflows in `notebook.ipynb`.

## Before you start

1. Search existing issues/PRs to avoid duplicates.
2. Open an issue for substantial changes and describe:
   - motivation,
   - expected impact,
   - proposed approach.

## Development setup

Use the environment described in `README.md`:

```bash
conda create -n dhaqnas python=3.10
conda activate dhaqnas
pip install torch==2.0.1 torchvision
pip install pennylane==0.33.1
pip install medmnist scikit-learn matplotlib pandas
```

## Branch and commit workflow

1. Fork the repository.
2. Create a feature branch from `main`.
3. Make focused commits with clear messages.
4. Open a pull request with a concise description.

## Pull request guidelines

Please include:

- **What changed** and **why**.
- Any relevant issue number (for example, `Closes #12`).
- Notes on reproducibility impact (datasets, hyperparameters, runtime, hardware).
- Updated documentation if behavior, usage, or results presentation changed.

For notebook-only changes:

- Keep cells organized and reproducible.
- Avoid committing unnecessary large outputs.
- Ensure paths and commands are portable.

## Validation

This repository is currently documentation/notebook-centric and does not define a dedicated automated CI test suite.

Before submitting:

- Ensure instructions in `README.md` still work.
- Re-run the modified notebook sections end-to-end where feasible.
- Confirm generated figures/tables are correctly labeled and referenced.

## Style and scope

- Keep changes minimal and focused.
- Do not mix unrelated refactors with functional changes.
- Preserve existing terminology and naming used in `README.md`.

## License

By contributing, you agree that your contributions are licensed under the repository's [MIT License](LICENSE).
