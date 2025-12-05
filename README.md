# SSTrans-SBIR (Reference Implementation)

Reference implementation for "SSTrans-SBIR: Spectral–Spatial Transformer for Fine-Grained Sketch-Based Image Retrieval".

**Important**: This repository provides a runnable reference implementation of the model described in the paper.
It is the user's responsibility to run the training scripts on appropriate hardware and datasets. 

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

📂 Datasets

- **ShoeV2 / ChairV2**  
  [Sketchy Official Website](https://sketchx.eecs.qmul.ac.uk/downloads/)  
  [Google Drive Download](https://drive.google.com/file/d/1frltfiEd9ymnODZFHYrbg741kfys1rq1/view)

- **Sketchy**  
  [Sketchy Official Website](https://sketchx.eecs.qmul.ac.uk/downloads/)  
  [Google Drive Download](https://drive.google.com/file/d/11GAr0jrtowTnR3otyQbNMSLPeHyvecdP/view)

- **TU-Berlin**  
  [TU-Berlin Official Website](https://www.tu-berlin.de/)  
  [Google Drive Download](https://drive.google.com/file/d/12VV40j5Nf4hNBfFy0AhYEtql1OjwXAUC/view)


Citation: If you use this code, please cite:

title = {Cross-Modal Spectral–Spatial Transformer for Fine-Grained SBIR},

author = {Mohammed A. S. Al-Mohamadi and Prabhakar C. J.},

journal = {arabian journal for science and engineering springer}, year = {2025} }

License: This project is released under the MIT License.

Contact: almohmdy30@gmail.com GitHub: https://github.com/mohammedalmohmdy
