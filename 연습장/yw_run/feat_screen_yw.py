"""피처 판정 A1·A2(FM)와 B(XGBoost) — 기록/피처판정기준_1125.md"""
import numpy as np, pandas as pd, json
from xgboost import XGBRegressor
B = "/home/user/fdi-team04-1-/연습장"; O = f"{B}/out_yw"
ALL = ["beta", "size", "bm", "rev", "mom", "ivol", "issue", "retail_nb", "foreign_nb"]
CAND = {"mom6": "+", "ltrev": "-", "high52": "+", "maxret": "-", "skew": "-", "season": "+", "consist": "+", "price": "+",
        "idvol": "-", "retail3": "-", "foreign3": "±", "bgrowth": "-"}
df = pd.read_parquet(f"{O}/features.parquet"); df = df[df.month >= "2001-01"].copy()
nw = pd.read_csv(f"{B}/out/목표가상향_변수_합치기용.csv", dtype={"code": str})
df = df.merge(nw[["month", "code", "news_missing", "tp_up"]], on=["month", "code"], how="left")
df["news_missing"] = df.news_missing.fillna(1); df["tp_up"] = df.tp_up.fillna(0)
df = df.merge(pd.read_parquet(f"{O}/feat_cand.parquet"), on=["month", "code"], how="left")
for c in ALL + list(CAND):
    df[c + "_r"] = df.groupby("month")[c].rank(pct=True).fillna(0.5)
for c in ALL: df[c + "_m"] = df[c].isna().astype(float)
df["y"] = df.groupby("month").target.rank(pct=True)
df.to_parquet(f"{O}/feat_panel.parquet")
BASE = [c + "_r" for c in ALL] + [c + "_m" for c in ("beta", "bm", "mom", "ivol", "issue")] + ["tp_up", "news_missing"]
def tt(x): x = pd.Series(x).dropna(); return x.mean() / (x.std() / np.sqrt(len(x)))
d = df[df.target.notna() & (df.month <= "2021-12")]
def fm(cols, last):
    B_ = {}
    for m, g in d.groupby("month"):
        X = np.column_stack([np.ones(len(g))] + [g[c] for c in cols]); B_[m] = np.linalg.lstsq(X, g.target * 100, rcond=None)[0][last]
    return pd.Series(B_)
ctrl = [c + "_r" for c in ALL]
XGB = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42, tree_method="hist", n_jobs=-1)
def xic(F):
    out = []
    for Y in range(2011, 2022):
        tr = df[(df.month <= f"{Y-1}-11") & df.y.notna()]; te = df[(df.month >= f"{Y}-01") & (df.month <= f"{Y}-12") & df.target.notna()]
        out.append(te[["month", "target"]].assign(s=XGBRegressor(**XGB).fit(tr[F], tr.y).predict(te[F])))
    p = pd.concat(out); return p.groupby("month").apply(lambda x: x.s.rank().corr(x.target.rank()), include_groups=False)
base = xic(BASE); base.to_csv(f"{O}/feat_base_ic.csv"); yb = base.groupby(base.index.str[:4]).mean()
rows = []
for c, sgn in list(CAND.items()) + [("dcd", "±")]:
    if c == "dcd":   # 거시 변수: 단면 순위를 못 바꿈 → 월별 베타 기울기를 dcd에 회귀
        bb = fm(ctrl, 1); x = d.groupby("month").dcd.first().reindex(bb.index)
        def ts(lo, hi):
            s = (bb.index >= lo) & (bb.index <= hi); X = np.column_stack([np.ones(s.sum()), x[s]]); yv = bb[s].values
            b, res, *_ = np.linalg.lstsq(X, yv, rcond=None); e = yv - X @ b; cov = (e @ e) / (len(yv) - 2) * np.linalg.inv(X.T @ X); return b[1], b[1] / np.sqrt(cov[1, 1])
        coef, t = ts("2001-01", "2021-12"); coef16, t16 = ts("2016-01", "2021-12"); name = "beta_cd (베타×금리변화)"
    else:
        s = fm(ctrl + [c + "_r"], -1); coef, t = s.mean(), tt(s); s16 = s[s.index >= "2016-01"]; coef16, t16 = s16.mean(), tt(s16); name = c
    a1 = abs(t) >= 2 and (sgn == "±" or (sgn == "+") == (coef > 0)); a2 = np.sign(coef16) == np.sign(coef)
    ic = xic(BASE + [c + "_r" if c != "dcd" else "dcd"]); yi = ic.groupby(ic.index.str[:4]).mean(); dd = ic - base
    b_ = dd.mean() > 0 and int((yi > yb).sum()) >= 6
    rows.append({"피처": name, "예상": sgn, "FM 계수": coef, "FM t": t, "2016-21 계수": coef16, "2016-21 t": t16, "A1": a1, "A2": a2,
                 "IC": ic.mean(), "IC 차이": dd.mean(), "차이 t": tt(dd), "오른 해": int((yi > yb).sum()), "B": b_})
    print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv(f"{O}/feat_screen.csv", index=False, encoding="utf-8-sig")
print("base IC", base.mean()); pd.set_option("display.width", 250); print(R.round(4).to_string(index=False))
