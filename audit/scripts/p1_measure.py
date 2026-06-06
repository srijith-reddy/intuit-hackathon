"""Part 1 gate measurement: shipped fold-σ vs cross-family-σ for A intervals.
Honest cross-fit coverage protocol (E2). READ-ONLY; decides whether to implement.
"""
import sys, os, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold, KFold
from src import data, features as F, models, calibration as cal
from src.config import SEED, set_seeds
Z = 1.6448536269514722

set_seeds()
tr, va = data.load_train(), data.load_validation()
X, art = F.build_features(tr, y=tr["default_flag"], groups=tr["business_id"])
lab = data.labeled_mask(tr); Xm = F.model_features(X)[lab.values]
y = tr.loc[lab, "default_flag"].astype(int).to_numpy(); g = tr.loc[lab, "business_id"]
Xva = F.model_features(F.build_features(va, fit_artifacts=art)[0])
vlab = data.labeled_mask(va).to_numpy()
yva = va.loc[vlab, "default_flag"].astype(int).to_numpy()
coh = data.assign_cohort_week(va).to_numpy()[vlab]

# ---- shipped model: lgb ensemble -> OOF-calibrated val point (honest) + fold-sigma
model, oof = models.train_pd_model(Xm, y, g)
mva, sva_fold = model.predict_raw(Xva)
raw_val = np.clip(mva[vlab], 0, 1)
# p_oof: each labeled-val point calibrated by isotonic fit on the OTHER val folds
# (realistic error, the honest proxy for test; using all-val iso would collapse alpha)
p_val = np.zeros(len(yva))
for tri, tei in KFold(5, shuffle=True, random_state=SEED).split(raw_val):
    isof = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw_val[tri], yva[tri])
    p_val[tei] = isof.transform(raw_val[tei])
fold_sigma = sva_fold[vlab]

# ---- cross-family sigma: train lgb/xgb/logit on IDENTICAL folds, train-OOF iso ---
splits = list(GroupKFold(5).split(Xm, y, g))
def fam_oof_and_val(kind):
    oof_ = np.zeros(len(Xm)); val_raw = np.zeros((vlab.sum(), 5))
    for j,(trn,vv) in enumerate(splits):
        if kind=="lgb":
            m=lgb.LGBMClassifier(n_estimators=700,learning_rate=.03,num_leaves=31,subsample=.8,
                colsample_bytree=.8,reg_lambda=1.,min_child_samples=50,random_state=SEED,verbose=-1)
            m.fit(Xm.iloc[trn],y[trn]); oof_[vv]=m.predict_proba(Xm.iloc[vv])[:,1]
            val_raw[:,j]=m.predict_proba(Xva.iloc[np.where(vlab)[0]])[:,1]
        elif kind=="xgb":
            m=xgb.XGBClassifier(n_estimators=600,learning_rate=.03,max_depth=5,subsample=.8,
                colsample_bytree=.8,reg_lambda=1.,eval_metric="logloss",random_state=SEED,n_jobs=-1)
            m.fit(Xm.iloc[trn],y[trn]); oof_[vv]=m.predict_proba(Xm.iloc[vv])[:,1]
            val_raw[:,j]=m.predict_proba(Xva.iloc[np.where(vlab)[0]])[:,1]
        else:
            m=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(C=.5,max_iter=2000))
            m.fit(Xm.iloc[trn],y[trn]); oof_[vv]=m.predict_proba(Xm.iloc[vv])[:,1]
            val_raw[:,j]=m.predict_proba(Xva.iloc[np.where(vlab)[0]])[:,1]
    iso=IsotonicRegression(out_of_bounds="clip",y_min=0,y_max=1).fit(oof_,y)
    return iso.transform(np.clip(val_raw.mean(1),0,1))   # train-OOF-calibrated val preds (leak-free wrt val)
fam = {k: fam_oof_and_val(k) for k in ["lgb","xgb","logit"]}
family_sigma = np.vstack([fam["lgb"],fam["xgb"],fam["logit"]]).std(0)

# ---- honest cross-fit coverage + gate metrics for a given sigma ------------------
def fit_alpha(p,s,yy):
    return cal.fit_pd_interval_scale(p, s, yy, n_bins=10)[0]
def gate(p, s, yy, cohort, name):
    # honest cross-fit coverage
    covs=[]
    for trn,tst in KFold(5,shuffle=True,random_state=SEED).split(p):
        a=fit_alpha(p[trn],s[trn],yy[trn])
        lo=np.clip(p[tst]-a*Z*s[tst],0,1); hi=np.clip(p[tst]+a*Z*s[tst],0,1)
        covs.append(cal.binned_coverage(p[tst],lo,hi,yy[tst],min(10,len(tst)//10 or 2)))
    honest=float(np.mean(covs))
    a=fit_alpha(p,s,yy)                       # ship alpha (full val)
    lo=np.clip(p-a*Z*s,0,1); hi=np.clip(p+a*Z*s,0,1); w=hi-lo
    # per-decile coverage
    dec=pd.qcut(p,10,duplicates="drop"); dd=pd.DataFrame({"p":p,"y":yy,"lo":lo,"hi":hi,"d":dec})
    dcov=dd.groupby("d",observed=True).apply(lambda gg:(gg.lo.mean()<=gg.y.mean()<=gg.hi.mean()))
    # per-cohort coverage
    cc=pd.DataFrame({"y":yy,"lo":lo,"hi":hi,"c":cohort}).groupby("c").apply(
        lambda gg:(gg.lo.mean()<=gg.y.mean()<=gg.hi.mean()))
    # width by error decile (adaptivity)
    err=np.abs(p-yy); edec=pd.qcut(err,10,duplicates="drop")
    wbyerr=pd.Series(w).groupby(np.asarray(edec),observed=True).mean()
    print(f"\n[{name}] alpha={a:.2f}  honest_xfit_coverage={honest:.3f}  mean_width={w.mean():.4f}")
    print(f"   per-decile coverage min={dcov.mean():.2f} ({int(dcov.sum())}/{len(dcov)})  "
          f"per-cohort min={cc.mean():.2f} ({int(cc.sum())}/{len(cc)})")
    print(f"   width adaptivity (low-err decile -> high-err decile): "
          f"{wbyerr.iloc[0]:.3f} ... {wbyerr.iloc[-1]:.3f}  ratio={wbyerr.iloc[-1]/max(wbyerr.iloc[0],1e-6):.2f}x")
    return dict(alpha=round(a,2),honest=round(honest,3),width=round(float(w.mean()),4),
                decile_cov=round(float(dcov.mean()),2),cohort_cov=round(float(cc.mean()),2),
                width_ratio=round(float(wbyerr.iloc[-1]/max(wbyerr.iloc[0],1e-6)),2))

print("="*60,"\nGATE COMPARISON: shipped fold-σ vs cross-family-σ\n",sep="")
r_fold = gate(p_val, fold_sigma, yva, coh, "SHIPPED fold-σ")
r_fam  = gate(p_val, family_sigma, yva, coh, "NEW cross-family-σ")
print("\n--- corr(σ,|err|) ---")
from scipy.stats import spearmanr
err=np.abs(p_val-yva)
print(f"   fold-σ:   {spearmanr(fold_sigma,err).correlation:+.3f}")
print(f"   family-σ: {spearmanr(family_sigma,err).correlation:+.3f}")
