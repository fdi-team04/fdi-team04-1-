"""목표가 순지표(상향 − 하향) 판정: 11/25 교수님 질문. 기준은 목표가 상향과 같음(A1 FM |t|≥2, A2 2016–21 방향, B XGBoost IC 상승+6/11년)"""
import numpy as np, pandas as pd
from xgboost import XGBRegressor
B = "/home/user/fdi-team04-1-/연습장"; O = f"{B}/out_yw"
ALL = ["beta", "size", "bm", "rev", "mom", "ivol", "issue", "retail_nb", "foreign_nb"]
f = pd.read_parquet(f"{O}/features.parquet"); nw = pd.read_csv(f"{B}/out/목표가상향_변수_합치기용.csv", dtype={"code": str}); nw["code"] = nw.code.str.zfill(6)
nw["tp_dn"] = (nw.n_tp_down > 0).astype(float); nw["tp_net"] = nw.tp_up - nw.tp_dn
def tt(x): x = pd.Series(x).dropna(); return x.mean() / (x.std() / np.sqrt(len(x)))
u = f.merge(nw[nw.news_missing == 0], on=["month", "code"]); u = u[u.target.notna() & (u.month >= "2001-01") & (u.month <= "2021-12")].copy()
for c in ALL: u[c + "_r"] = u.groupby("month")[c].rank(pct=True).fillna(0.5)
def fm(v, lo, hi):
    Bs = []
    for m, g in u[(u.month >= lo) & (u.month <= hi)].groupby("month"):
        X = np.column_stack([np.ones(len(g))] + [g[c + "_r"] for c in ALL] + [g[x] for x in v]); Bs.append(np.linalg.lstsq(X, g.target * 100, rcond=None)[0][-len(v):])
    return np.array(Bs)
print("하향 이벤트", int(u.tp_dn.sum()), "상향", int(u.tp_up.sum()), "둘 다", int(((u.tp_up == 1) & (u.tp_dn == 1)).sum()))
rows = []
for lab, v in [("하향만", ["tp_dn"]), ("순지표(상향−하향)", ["tp_net"]), ("상향+하향 따로", ["tp_up", "tp_dn"])]:
    a = fm(v, "2001-01", "2021-12"); b = fm(v, "2016-01", "2021-12")
    for i, x in enumerate(v):
        rows.append({"모델": lab, "변수": x, "FM 계수": a[:, i].mean(), "t": tt(a[:, i]), "2016-21": b[:, i].mean(), "2016-21 t": tt(b[:, i])})
R = pd.DataFrame(rows); print(R.round(3).to_string(index=False)); R.to_csv(f"{O}/tp_net_fm.csv", index=False, encoding="utf-8-sig")
# B: XGBoost 전체 모델
df = pd.read_parquet(f"{O}/feat_panel.parquet").merge(nw[["month", "code", "tp_dn", "tp_net"]], on=["month", "code"], how="left")
df["tp_dn"] = df.tp_dn.fillna(0); df["tp_net"] = df.tp_net.fillna(0)
BASE = [c + "_r" for c in ALL] + [c + "_m" for c in ("beta", "bm", "mom", "ivol", "issue")] + ["news_missing"]
XGB = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42, tree_method="hist", n_jobs=-1)
def xic(F):
    out = []
    for Y in range(2011, 2022):
        tr = df[(df.month <= f"{Y-1}-11") & df.y.notna()]; te = df[(df.month >= f"{Y}-01") & (df.month <= f"{Y}-12") & df.target.notna()]
        out.append(te[["month", "target"]].assign(s=XGBRegressor(**XGB).fit(tr[F], tr.y).predict(te[F])))
    p = pd.concat(out); return p.groupby("month").apply(lambda x: x.s.rank().corr(x.target.rank()), include_groups=False)
base = xic([c for c in BASE if c != "news_missing"]); yb = base.groupby(base.index.str[:4]).mean(); rows = []
for lab, F in [("상향(현재)", BASE + ["tp_up"]), ("순지표", BASE + ["tp_net"]), ("상향+하향", BASE + ["tp_up", "tp_dn"])]:
    i = xic(F); yi = i.groupby(i.index.str[:4]).mean()
    rows.append({"모델": lab, "IC": i.mean(), "정형 9개 대비": (i - base).mean(), "차이 t": tt(i - base), "오른 해": int((yi > yb).sum())}); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f"{O}/tp_net_xgb.csv", index=False, encoding="utf-8-sig"); print("정형 9개 IC", base.mean())
