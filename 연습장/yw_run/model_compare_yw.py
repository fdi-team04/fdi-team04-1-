"""XGBoost vs 다른 비선형 모델 비교 (같은 조건: 정형 9개 + 목표가 상향, 확장 학습 2011–2021, seed 42, 설정은 기본값 근처로 고정)
판단 기준(미리 정함): XGBoost보다 평균 IC가 높고 11년 중 6년 이상 이기면 '더 좋다'
데이터: v2 이전 정형 데이터(out_yw/features.parquet) — 처음 결과와 같은 데이터
"""
import numpy as np, pandas as pd, time
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
B = "/home/user/fdi-team04-1-/연습장"; OUT = f"{B}/out_yw"
ALL = ["beta", "size", "bm", "rev", "mom", "ivol", "issue", "retail_nb", "foreign_nb"]
df = pd.read_parquet(f"{OUT}/features.parquet"); df = df[df.month >= "2001-01"].copy()
nw = pd.read_csv(f"{B}/out/목표가상향_변수_합치기용.csv", dtype={"code": str})
df = df.merge(nw[["month", "code", "news_missing", "tp_up"]], on=["month", "code"], how="left")
df["news_missing"] = df.news_missing.fillna(1); df["tp_up"] = df.tp_up.fillna(0)
for c in ALL:
    df[c + "_m"] = df[c].isna().astype(float); df[c + "_r"] = df.groupby("month")[c].rank(pct=True).fillna(0.5)
df["y"] = df.groupby("month").target.rank(pct=True)
F = [c + "_r" for c in ALL] + [c + "_m" for c in ("beta", "bm", "mom", "ivol", "issue")] + ["tp_up", "news_missing"]
MODELS = {
 "XGBoost (현재)": lambda: XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42, tree_method="hist", n_jobs=-1),
 "LightGBM": lambda: LGBMRegressor(n_estimators=300, num_leaves=15, learning_rate=0.05, subsample=0.8, subsample_freq=1, random_state=42, n_jobs=-1, verbose=-1),
 "랜덤포레스트": lambda: RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=200, max_features=0.5, random_state=42, n_jobs=-1),
 "신경망 (MLP)": lambda: MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-3, learning_rate_init=1e-3, max_iter=100, early_stopping=True, random_state=42),
 "릿지 (선형, 참고)": lambda: Ridge(alpha=1.0),
}
def tt(x): return x.mean() / (x.std() / np.sqrt(len(x)))
ics = {}
for name, mk in MODELS.items():
    t0 = time.time(); out = []
    for Y in range(2011, 2022):
        tr = df[(df.month <= f"{Y-1}-11") & df.y.notna()]; te = df[(df.month >= f"{Y}-01") & (df.month <= f"{Y}-12") & df.target.notna()]
        out.append(te[["month", "target"]].assign(score=mk().fit(tr[F], tr.y).predict(te[F])))
    p = pd.concat(out)
    ics[name] = p.groupby("month").apply(lambda x: x.score.rank().corr(x.target.rank()), include_groups=False)
    print(f"{name}: IC {ics[name].mean():.4f} ({time.time()-t0:.0f}s)", flush=True)
base = ics["XGBoost (현재)"]; by = pd.DataFrame({k: v.groupby(v.index.str[:4]).mean() for k, v in ics.items()})
rows = []
for k, v in ics.items():
    d = v - base
    rows.append({"모델": k, "평균 IC": v.mean(), "IC t": tt(v), "XGBoost 대비": d.mean(), "차이 t": tt(d) if k != "XGBoost (현재)" else np.nan,
                 "XGBoost 이긴 해(/11)": int((by[k] > by["XGBoost (현재)"]).sum()) if k != "XGBoost (현재)" else np.nan})
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/model_compare.csv", index=False, encoding="utf-8-sig"); by.to_csv(f"{OUT}/model_compare_by_year.csv", encoding="utf-8-sig")
pd.set_option("display.width", 200); print(R.round(4).to_string(index=False)); print(by.round(3).to_string())
