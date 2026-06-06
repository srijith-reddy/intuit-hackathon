"""Part 2: model-family parity + logit coefficient interpretability. READ-ONLY.
Produces the comparison table and the logit coefficient/SHAP sign-agreement check.
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
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from src import data, features as F
from src.config import SEED, set_seeds

set_seeds()
tr, va = data.load_train(), data.load_validation()
X, art = F.build_features(tr, y=tr["default_flag"], groups=tr["business_id"])
lab = data.labeled_mask(tr); Xm = F.model_features(X)[lab.values]
y = tr.loc[lab, "default_flag"].astype(int).to_numpy(); g = tr.loc[lab, "business_id"]
Xva = F.model_features(F.build_features(va, fit_artifacts=art)[0])
vlab = data.labeled_mask(va).to_numpy(); yva = va.loc[vlab, "default_flag"].astype(int).to_numpy()
cohv = data.assign_cohort_week(va).to_numpy()[vlab]

def ece(p, yy, b=10):
    d=pd.DataFrame({"p":p,"y":yy}); d["bin"]=pd.qcut(p,b,duplicates="drop")
    gg=d.groupby("bin",observed=True).agg(pm=("p","mean"),ym=("y","mean"),n=("y","size"))
    return float((gg.n*(gg.pm-gg.ym).abs()).sum()/gg.n.sum())

splits=list(GroupKFold(5).split(Xm,y,g))
def run(kind):
    oof=np.zeros(len(Xm)); vp=np.zeros((vlab.sum(),5))
    for j,(trn,vv) in enumerate(splits):
        if kind=="lgb": m=lgb.LGBMClassifier(n_estimators=700,learning_rate=.03,num_leaves=31,subsample=.8,colsample_bytree=.8,reg_lambda=1.,min_child_samples=50,random_state=SEED,verbose=-1)
        elif kind=="xgb": m=xgb.XGBClassifier(n_estimators=600,learning_rate=.03,max_depth=5,subsample=.8,colsample_bytree=.8,reg_lambda=1.,eval_metric="logloss",random_state=SEED,n_jobs=-1)
        else: m=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(C=.5,max_iter=2000))
        m.fit(Xm.iloc[trn],y[trn]); oof[vv]=m.predict_proba(Xm.iloc[vv])[:,1]
        vp[:,j]=m.predict_proba(Xva.iloc[np.where(vlab)[0]])[:,1]
    iso=IsotonicRegression(out_of_bounds="clip",y_min=0,y_max=1).fit(oof,y)
    vraw=vp.mean(1); vcal=iso.transform(np.clip(vraw,0,1))
    # per-cohort worst val AUC
    wc=1.0
    for c in np.unique(cohv):
        mc=cohv==c
        if mc.sum()>20 and len(np.unique(yva[mc]))==2: wc=min(wc,roc_auc_score(yva[mc],vraw[mc]))
    return dict(oof_auc=roc_auc_score(y,oof),oof_brier=brier_score_loss(y,oof),
                val_auc=roc_auc_score(yva,vraw),val_ece=ece(vcal,yva),worst_cohort_auc=wc), oof
res={}; oofs={}
for k in ["lgb","xgb","logit"]: res[k],oofs[k]=run(k)
bl=(oofs["lgb"]+oofs["xgb"]+oofs["logit"])/3
res["blend"]=dict(oof_auc=roc_auc_score(y,bl),oof_brier=brier_score_loss(y,bl),val_auc=np.nan,val_ece=np.nan,worst_cohort_auc=np.nan)
print("=== model family comparison ===")
print(pd.DataFrame(res).T.round(4).to_string())

# ---- logit standardized coefficients + sign agreement -----------------------
imp=SimpleImputer(strategy="median"); sc=StandardScaler()
Xs=sc.fit_transform(imp.fit_transform(Xm))
lr=LogisticRegression(C=.5,max_iter=2000).fit(Xs,y)
coef=pd.Series(lr.coef_[0],index=Xm.columns)
top=coef.reindex(coef.abs().sort_values(ascending=False).index).head(10)
# univariate corr(feature,y) as a direction proxy; registry expected sign
reg=F.REGISTRY
print("\n=== logit top-10 standardized coefficients (sign agreement) ===")
rows=[]
for f,c in top.items():
    xv=pd.to_numeric(Xm[f],errors="coerce")
    corr=np.corrcoef(xv.fillna(xv.median()),y)[0,1]
    exp=reg[f].expected_sign if f in reg else "?"
    logit_sign="+" if c>0 else "-"; corr_sign="+" if corr>0 else "-"
    agree="OK" if logit_sign==corr_sign else "CONFLICT"
    rows.append((f,round(c,3),logit_sign,round(corr,3),corr_sign,exp,agree))
cmp=pd.DataFrame(rows,columns=["feature","logit_coef","logit_sign","corr_y","corr_sign","registry_exp","logit_vs_corr"])
print(cmp.to_string(index=False))
print("\nsign conflicts (logit vs univariate-corr):", int((cmp.logit_vs_corr=="CONFLICT").sum()))
