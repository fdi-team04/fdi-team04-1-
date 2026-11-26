"""후보 피처의 기간 안정성: 2011–16 vs 2017–21 (XGBoost). 진짜 검증은 2022년 이후 — 이건 앞뒤 기간 일관성 확인"""
import numpy as np, pandas as pd
from xgboost import XGBRegressor
O = "/home/user/fdi-team04-1-/연습장/out_yw"; df = pd.read_parquet(f"{O}/feat_panel.parquet")
ALL = ["beta", "size", "bm", "rev", "mom", "ivol", "issue", "retail_nb", "foreign_nb"]
BASE = [c + "_r" for c in ALL] + [c + "_m" for c in ("beta", "bm", "mom", "ivol", "issue")] + ["tp_up", "news_missing"]
XGB = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42, tree_method="hist", n_jobs=-1)
def xic(F):
    out = []
    for Y in range(2011, 2022):
        tr = df[(df.month <= f"{Y-1}-11") & df.y.notna()]; te = df[(df.month >= f"{Y}-01") & (df.month <= f"{Y}-12") & df.target.notna()]
        out.append(te[["month", "target"]].assign(s=XGBRegressor(**XGB).fit(tr[F], tr.y).predict(te[F])))
    p = pd.concat(out); return p.groupby("month").apply(lambda x: x.s.rank().corr(x.target.rank()), include_groups=False)
def tt(x): return x.mean() / (x.std() / np.sqrt(len(x)))
b = pd.read_csv(f"{O}/feat_base_ic.csv", index_col=0).iloc[:, 0]; yb = b.groupby(b.index.str[:4]).mean(); rows = []
for lab, F in [("idvol", ["idvol_r"]), ("mom6", ["mom6_r"]), ("high52", ["high52_r"]), ("ltrev", ["ltrev_r"]), ("maxret", ["maxret_r"]), ("bgrowth", ["bgrowth_r"]),
               ("idvol+mom6", ["idvol_r", "mom6_r"]), ("idvol+mom6+high52", ["idvol_r", "mom6_r", "high52_r"])]:
    i = xic(BASE + F); d = i - b; yi = i.groupby(i.index.str[:4]).mean()
    for per, lo, hi in [("2011–16", "2011", "2016-12"), ("2017–21", "2017", "2021-12")]:
        k = (d.index >= lo) & (d.index <= hi); yy = (yi > yb)[(yi.index >= lo[:4]) & (yi.index <= hi[:4])]
        rows.append({"피처": lab, "기간": per, "IC 차이": d[k].mean(), "t": tt(d[k]), "오른 해": f"{int(yy.sum())}/{len(yy)}"})
    print(rows[-2:], flush=True)
R = pd.DataFrame(rows); R.to_csv(f"{O}/feat_half.csv", index=False, encoding="utf-8-sig"); print(R.round(4).to_string(index=False))
