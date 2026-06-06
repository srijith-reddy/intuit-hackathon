"""Probe: does a diverse LGBM+XGB+logistic ensemble help, and WHERE? READ-ONLY.
Compares OOF AUC/Brier of each base model vs a blend, and checks whether multi-family
disagreement is a better epistemic-uncertainty signal than fold disagreement (for the
20% calibration component). Does not touch the shipped pipeline.
"""
import sys, os, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from src import data, features as F
from src.config import SEED, set_seeds

set_seeds()
tr = data.load_train(); y_all = tr["default_flag"]
X, art = F.build_features(tr, y=y_all, groups=tr["business_id"])
lab = data.labeled_mask(tr)
Xm = F.model_features(X)[lab.values]
y = tr.loc[lab, "default_flag"].astype(int).to_numpy()
g = tr.loc[lab, "business_id"]
Ximp = Xm.fillna(Xm.median())  # logistic needs dense

oof = {k: np.zeros(len(Xm)) for k in ["lgb", "xgb", "logit"]}
for trn, val in GroupKFold(5).split(Xm, y, g):
    m1 = lgb.LGBMClassifier(n_estimators=700, learning_rate=0.03, num_leaves=31, subsample=.8,
                            colsample_bytree=.8, reg_lambda=1.0, min_child_samples=50,
                            random_state=SEED, verbose=-1)
    m1.fit(Xm.iloc[trn], y[trn]); oof["lgb"][val] = m1.predict_proba(Xm.iloc[val])[:, 1]
    m2 = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=5, subsample=.8,
                           colsample_bytree=.8, reg_lambda=1.0, eval_metric="logloss",
                           random_state=SEED, n_jobs=-1)
    m2.fit(Xm.iloc[trn], y[trn]); oof["xgb"][val] = m2.predict_proba(Xm.iloc[val])[:, 1]
    m3 = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    m3.fit(Ximp.iloc[trn], y[trn]); oof["logit"][val] = m3.predict_proba(Ximp.iloc[val])[:, 1]

print("=== OOF base models (GroupKFold) ===")
for k, p in oof.items():
    print(f"  {k:6s} AUC={roc_auc_score(y,p):.4f}  Brier={brier_score_loss(y,p):.4f}")
blend = (oof["lgb"] + oof["xgb"] + oof["logit"]) / 3
blend_lx = (oof["lgb"] + oof["xgb"]) / 2
print(f"  blend(LGB+XGB+logit) AUC={roc_auc_score(y,blend):.4f}  Brier={brier_score_loss(y,blend):.4f}")
print(f"  blend(LGB+XGB)       AUC={roc_auc_score(y,blend_lx):.4f}  Brier={brier_score_loss(y,blend_lx):.4f}")
print(f"  current shipped (LGB only) AUC={roc_auc_score(y,oof['lgb']):.4f}  Brier={brier_score_loss(y,oof['lgb']):.4f}")

print("\n=== uncertainty signal: does cross-family disagreement track error? ===")
# multi-family std vs |error|; a good epistemic signal correlates with realized error
fam_std = np.vstack([oof['lgb'], oof['xgb'], oof['logit']]).std(0)
err = np.abs(blend - y)
from scipy.stats import spearmanr
print(f"  corr(cross-family std, |error|)   = {spearmanr(fam_std, err).correlation:+.3f}")
# compare to lgb-only single number (no fold std available here) -> proxy: lgb dist from blend
print(f"  cross-family std: mean={fam_std.mean():.4f}  p90={np.quantile(fam_std,.9):.4f}")
print(f"  does std widen on hard cases? mean std in top-error decile vs bottom: "
      f"{fam_std[err>=np.quantile(err,.9)].mean():.4f} vs {fam_std[err<=np.quantile(err,.1)].mean():.4f}")
