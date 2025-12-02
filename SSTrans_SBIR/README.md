# SSTrans-SBIR (Reference Implementation)

Reference implementation for "SSTrans-SBIR: Spectral–Spatial Transformer for Fine-Grained Sketch-Based Image Retrieval".

**Important**: This repository provides a runnable reference implementation of the model described in the paper.
It is the user's responsibility to run the training scripts on appropriate hardware and datasets. Results reported
in the paper should be reproduced by running the experiments; if you (the author) already ran experiments,
add the trained checkpoints and logs under `checkpoints/` and update `RESULTS.md`.

## Repository structure
- `model.py`         : model definitions (DCT branch, CNN/ViT backbone, SSF, CDT, projection)
- `dataset.py`       : dataset loaders (Chair-V2, Sketchy, TU-Berlin) — fill paths
- `losses.py`        : triplet loss + alignment loss
- `train.py`         : training loop
- `eval.py`          : evaluation metrics (CMC, Precision@K, mAP)
- `visualize.py`     : attention heatmaps and retrieval visualization
- `utils.py`         : helpers (save/load, metrics)
- `requirements.txt` : Python packages
- `Dockerfile`       : optional container
- `configs/`         : example YAML configs
- `checkpoints/`     : (user) place for saved models
- `RESULTS.md`       : (user) add reproduced tables & logs

## Quick start (example)
1. Create Python env:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Prepare datasets:
   - Place Chair-V2 / Sketchy / TU-Berlin images in folders and update paths in `dataset.py` or `configs/example.yaml`.

3. Train:
   ```bash
   python train.py --config configs/chairv2_train.yaml --save_dir checkpoints/chairv2_run1
   ```

4. Evaluate:
   ```bash
   python eval.py --checkpoint checkpoints/chairv2_run1/epoch_30.pth --data_root /path/to/chairv2
   ```

5. Visualize attention:
   ```bash
   python visualize.py --checkpoint checkpoints/chairv2_run1/epoch_30.pth --sketch examples/sketch1.png
   ```

## Reproducibility
- Seeds: set random seeds via `--seed` flag in `train.py`.
- Hardware: GPU recommended (NVIDIA, with CUDA >=11.1).
- If you include pre-trained checkpoints, add them to `checkpoints/` and provide provenance (training logs, exact command lines).

## License
MIT License (see LICENSE)
