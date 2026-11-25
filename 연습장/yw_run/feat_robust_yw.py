"""판정 C: B 통과 피처를 LightGBM·랜덤포레스트·신경망에서 확인 + 묶음 모델 5개 비교"""
import numpy as np, pandas as pd, time, sys
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
O = "/home/user/fdi-team04-1-/연습장/out_yw"
df = pd.read_parquet(f"{O}/feat_panel.parquet")
ALL = ["beta", "size", "bm", "rev", "mom", "ivol", "issue", "retail_nb", "foreign_nb"]
BASE = [c + "_r" for c in ALL] + [c + "_m" for c in ("beta", "bm", "mom", "ivol", "issue")] + ["tp_up", "news_missing"]
PASS = ["mom6", "ltrev", "high52", "maxret", "skew", "idvol", "bgrowth"]
M = {"LightGBM": lambda: LGBMRegressor(n_estimators=300, num_leaves=15, learning_rate=0.05, subsample=0.8, subsample_freq=1, random_state=42, n_jobs=-1, verbose=-1),
     "랜덤포레스트": lambda: RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=200, max_features=0.5, random_state=42, n_jobs=-1),
     "신경망": lambda: MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=100, early_stopping=True, random_state=42),
     "XGBoost": lambda: XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42, tree_method="hist", n_jobs=-1),
     "릿지": lambda: Ridge(alpha=1.0)}
def ic(mk, F):
    out = []
    for Y in range(2011, 2022):
        tr = df[(df.month <= f"{Y-1}-11") & df.y.notna()]; te = df[(df.month >= f"{Y}-01") & (df.month <= f"{Y}-12") & df.target.notna()]
        out.append(te[["month", "target"]].assign(s=mk().fit(tr[F], tr.y).predict(te[F])))
    p = pd.concat(out); return p.groupby("month").apply(lambda x: x.s.rank().corr(x.target.rank()), include_groups=False)
def tt(x): return x.mean() / (x.std() / np.sqrt(len(x)))
rows = []
for mname in ["LightGBM", "랜덤포레스트", "신경망"]:
    b = ic(M[mname], BASE); yb = b.groupby(b.index.str[:4]).mean()
    for c in PASS:
        t0 = time.time(); i = ic(M[mname], BASE + [c + "_r"]); yi = i.groupby(i.index.str[:4]).mean()
        rows.append({"모델": mname, "피처": c, "기준 IC": b.mean(), "IC": i.mean(), "차이": (i - b).mean(), "차이 t": tt(i - b), "오른 해": int((yi > yb).sum())})
        print(rows[-1], f"{time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(f"{O}/feat_robust.csv", index=False, encoding="utf-8-sig")
# 묶음 모델: (1) B 통과 7개 전부 (2) idvol만
rows2 = []
for mname, mk in M.items():
    b = ic(mk, BASE); yb = b.groupby(b.index.str[:4]).mean()
    for lab, F in [("+B통과 7개", BASE + [c + "_r" for c in PASS]), ("+idvol", BASE + ["idvol_r"]), ("+mom6+idvol", BASE + ["mom6_r", "idvol_r"])]:
        i = ic(mk, F); yi = i.groupby(i.index.str[:4]).mean()
        rows2.append({"모델": mname, "묶음": lab, "기준 IC": b.mean(), "IC": i.mean(), "차이": (i - b).mean(), "차이 t": tt(i - b), "오른 해": int((yi > yb).sum())})
        print(rows2[-1], flush=True)
    pd.DataFrame(rows2).to_csv(f"{O}/feat_bundle.csv", index=False, encoding="utf-8-sig")
print("DONE")
