"""모델을 바꿔도 목표가 상향이 IC를 올리나: 랜덤포레스트·신경망·LightGBM에서 기준(정형 9개) vs +목표가 상향"""
import importlib.util, sys, numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location("mc", "/home/user/fdi-team04-1-/연습장/yw_run/model_compare_yw.py")
src = open(spec.origin, encoding="utf-8").read().split("def tt(x)")[0]; ns = {}; exec(src, ns)
df, F, MODELS = ns["df"], ns["F"], ns["MODELS"]; BASE = [f for f in F if f not in ("tp_up", "news_missing")]
def tt(x): return x.mean() / (x.std() / np.sqrt(len(x)))
def run(mk, cols):
    out = []
    for Y in range(2011, 2022):
        tr = df[(df.month <= f"{Y-1}-11") & df.y.notna()]; te = df[(df.month >= f"{Y}-01") & (df.month <= f"{Y}-12") & df.target.notna()]
        out.append(te[["month", "target"]].assign(score=mk().fit(tr[cols], tr.y).predict(te[cols])))
    p = pd.concat(out); return p.groupby("month").apply(lambda x: x.score.rank().corr(x.target.rank()), include_groups=False)
full = pd.read_csv("/home/user/fdi-team04-1-/연습장/out_yw/model_compare.csv")
rows = []
for k in ["LightGBM", "랜덤포레스트", "신경망 (MLP)"]:
    b = run(MODELS[k], BASE); t = run(MODELS[k], F)
    yb, yt = b.groupby(b.index.str[:4]).mean(), t.groupby(t.index.str[:4]).mean()
    rows.append({"모델": k, "정형 9개 IC": b.mean(), "+목표가 IC": t.mean(), "차이": (t - b).mean(), "차이 t": tt(t - b), "오른 해(/11)": int((yt > yb).sum())})
    print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv("/home/user/fdi-team04-1-/연습장/out_yw/model_compare_tp.csv", index=False, encoding="utf-8-sig"); print(R.round(4).to_string(index=False))
