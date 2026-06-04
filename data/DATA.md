# Data Preparation

Defaults in [`tokengs/data/registry.py`](../tokengs/data/registry.py) resolve dataset roots under the **repository root**:

| Path | Role |
|------|------|
| `data/dl3dv` | Training zips for `DL3DV10K` (e.g. DL3DV-ALL 960p undistorted). |
| `data/dl3dv_eval` | Eval set for `DL3DVEval` (e.g. DL3DV-10K-Benchmark). |
| `data/kubric` | Kubric multi-view 4D tar dump used by `finetune_dl3dv_kubric_*` presets. |

Implementation details for readers and transforms live in [`tokengs/data/static/dl3dv.py`](../tokengs/data/static/dl3dv.py).

## Symlinks (recommended)

From the **repository root**:

```bash
mkdir -p data
ln -snf /absolute/path/to/DL3DV-ALL-960P-undistorted data/dl3dv
ln -snf /absolute/path/to/DL3DV-10K-Benchmark data/dl3dv_eval
ln -snf /absolute/path/to/objaverse_4d/kubric_mv data/kubric
```

`-snf` creates or replaces a symlink. Relative targets (e.g. `../datasets/dl3dv`) are fine if paths stay stable.

## Kubric Layout

The Kubric reader expects split directories and per-camera tar files:

```text
data/kubric/
  v0/
    <scene>/
      output_000.tar
      output_001.tar
      ...
  v1/
  v2/
```

Each `output_{view:03d}.tar` should contain `metadata.json`, `rgba_{frame:05d}.png`, and `depth_{frame:05d}.tiff`. The dynamic presets enable pointmap camera scaling, so the depth TIFFs are loaded for the input frames.

## Overrides without symlinks

Pass kwargs the dataset constructor accepts (for example `root_path`) via Tyro. See `dataset_kwargs` on [`Options`](../tokengs/options.py) and run:

```bash
python -m tokengs.train --help
python -m tokengs.evaluate --help
```
