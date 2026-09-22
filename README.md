# causal-tns

Code and experiments for the research program described in
[`docs/causal_tn_notes.tex`](docs/causal_tn_notes.tex).

## Environment

```bash
conda env create -f environment.yml
conda activate causal-tns
```

To update dependencies after changing `environment.yml`:

```bash
conda env update -n causal-tns -f environment.yml --prune
```

Reusable code belongs in `src/`, and experiments belong in `experiments/`.
