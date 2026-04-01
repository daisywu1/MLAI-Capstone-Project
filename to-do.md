# Next Steps — After auth.txt.gz Download Completes

## 1. Move the file into the project

```bash
cp ~/Downloads/auth.txt.gz ~/gh-ws/MLAI-Capstone-Project/data/raw/
```

## 2. Verify it arrived

```bash
ls -lh ~/gh-ws/MLAI-Capstone-Project/data/raw/
```

You should see:
- `auth.txt.gz` (~7.2 GB)
- `redteam.txt.gz` (~4.8 KB)

## 3. Activate the virtual environment

```bash
cd ~/gh-ws/MLAI-Capstone-Project
source venv/bin/activate
```

## 4. Launch the notebook

```bash
cd notebooks
jupyter notebook capstone_analysis.ipynb
```

## 5. Select the correct kernel

In Jupyter, go to **Kernel → Change Kernel → Python 3 (Capstone)**

## 6. Run All Cells

**Kernel → Restart & Run All**

- Cell 1 (imports): ~10 seconds
- Cell 2 (load 1M rows): ~1–2 minutes
- Cell 3 (feature engineering): ~3–5 minutes
- Remaining cells (models, plots): ~5–10 minutes
- **Total: ~15–20 minutes**

## 7. Check outputs

- `images/` — all saved plots (PNG)
- `data/processed/` — cleaned parquet files
- `models/` — saved trained models

## Done!
