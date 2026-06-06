"""SHAP explanation of the PD model — backs the writeup's driver claims.

`python -m src.explain` fits the PD model on labeled train and writes:
  - reports/figures/shap_summary.png   (beeswarm over the model features)
  - reports/shap_top_features.csv      (mean |SHAP| ranking)
Used so Deliverable D's "drivers explained with SHAP" is a real, reproducible artifact.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src import data, features as F, models
from src.config import PATHS, SEED, set_seeds

FIGDIR = PATHS.reports / "figures"


def compute(sample: int = 4000):
    set_seeds()
    tr = data.load_train()
    X, art = F.build_features(tr, y=tr["default_flag"], groups=tr["business_id"])
    lab = data.labeled_mask(tr)
    Xm = F.model_features(X)[lab.values]
    y = tr.loc[lab, "default_flag"].astype(int).to_numpy()
    model, _ = models.train_pd_model(Xm, y, tr.loc[lab, "business_id"])

    # SHAP over a sample, averaged across the fold ensemble (faithful to what we ship)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Xm), size=min(sample, len(Xm)), replace=False)
    Xs = Xm.iloc[idx]
    shap_acc = np.zeros(Xs.shape)
    for b in model.boosters:
        ex = shap.TreeExplainer(b)
        sv = ex.shap_values(Xs)
        sv = sv[1] if isinstance(sv, list) else sv  # positive class
        shap_acc += np.asarray(sv)
    shap_vals = shap_acc / len(model.boosters)

    mean_abs = pd.Series(np.abs(shap_vals).mean(0), index=Xs.columns).sort_values(ascending=False)
    mean_abs.to_csv(PATHS.reports / "shap_top_features.csv", header=["mean_abs_shap"])

    plt.figure()
    shap.summary_plot(shap_vals, Xs, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(FIGDIR / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    return mean_abs


if __name__ == "__main__":
    m = compute()
    print("Top 12 drivers by mean |SHAP|:")
    print(m.head(12).round(4).to_string())
    print("\nWrote reports/figures/shap_summary.png + reports/shap_top_features.csv")
