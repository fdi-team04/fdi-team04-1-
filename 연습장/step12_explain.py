"""06 설명 단계: XGBoost 안의 방향(SHAP) vs Fama-MacBeth 방향, 그리고 두 모델의 예측값 비교

왜 하나 (11/11 교수님 피드백 F11 + 이안 제안):
 - "트리 모델은 +/−를 모르지만 SHAP 값을 찍으면 얼추 볼 수 있다"
 - "06 파일 마저 작성해서 파마 맥베스가 예상하는 수치랑 XGBoost가 예상하는 수치 차이 보고"
무엇을 하나:
 A. SHAP: 수급까지 넣은 XGBoost(학습 2001-01~2020-11, 2021년 예측에 쓴 마지막 모델)로
    검증 구간(2011~2021) 종목-월 2만 개의 SHAP 값을 계산.
    - 중요도 = 평균 |SHAP|
    - 방향 = 변수 순위(0~1)와 SHAP 값의 상관 (+면 "클수록 예측을 올림")
    - 모양 = 변수 10분위별 평균 SHAP (dependence)
 B. FM 예측: 매년 1월에 "그때까지의 월별 FM 계수 평균"으로 그해 12개월 예측 (XGBoost와 같은 누적 방식)
    → FM 예측 IC vs XGBoost IC, 그리고 두 예측의 순위상관(얼마나 같은 종목을 고르나)
"""
import numpy as np, pandas as pd, matplotlib, sys
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
FD = "/tmp/claude-0/-home-user-fdi-team04-1-/03b15712-f400-5188-9b23-fea64cb12856/scratchpad/fonts"
for f in ("plex400.ttf", "plex600.ttf"): fm.fontManager.addfont(f"{FD}/{f}")
plt.rcParams.update({"font.family": "IBM Plex Sans KR", "axes.unicode_minus": False, "font.size": 14,
                     "axes.spines.top": False, "axes.spines.right": False})
from xgboost import XGBRegressor
import shap
OUT = "/home/user/fdi-team04-1-/연습장/out"
ALL = ["beta", "size", "bm", "rev", "mom", "ivol", "issue", "retail_nb", "foreign_nb"]
NM = {"beta": "베타", "size": "규모", "bm": "가치", "rev": "단기 반전", "mom": "모멘텀", "ivol": "고유변동성",
      "issue": "주식 발행", "retail_nb": "개인 순매수", "foreign_nb": "외국인 순매수"}
XGB = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42, tree_method="hist", n_jobs=-1)

df = pd.read_parquet(f"{OUT}/features.parquet"); df = df[df.month >= "2001-01"].copy()
for c in ALL:
    df[c + "_m"] = df[c].isna().astype(float)
    df[c + "_r"] = df.groupby("month")[c].rank(pct=True).fillna(0.5)
df["y"] = df.groupby("month").target.rank(pct=True)
F = [c + "_r" for c in ALL] + [c + "_m" for c in ("beta", "bm", "mom", "ivol", "issue")]

# ---------- A. SHAP ----------
tr = df[(df.month <= "2020-11") & df.y.notna()]
mdl = XGBRegressor(**XGB).fit(tr[F], tr.y)
va = df[(df.month >= "2011-01") & df.target.notna()].sample(20000, random_state=42)
sv = shap.TreeExplainer(mdl).shap_values(va[F])
S = pd.DataFrame(sv, columns=F, index=va.index)
rows = []
for c in ALL:
    r = c + "_r"
    rows.append({"col": c, "변수": NM[c], "중요도(평균|SHAP|)": S[r].abs().mean(),
                 "SHAP 방향(순위~SHAP 상관)": np.corrcoef(va[r], S[r])[0, 1],
                 "하위10% 평균SHAP": S[r][va[r] <= 0.1].mean(), "상위10% 평균SHAP": S[r][va[r] >= 0.9].mean()})
A = pd.DataFrame(rows)
fmres = pd.read_csv(f"{OUT}/fama_macbeth.csv"); fmres = fmres[fmres.구간 == "검증 2011~2021"].set_index("col")
A["FM 계수(%p)"] = A.col.map(fmres["계수(%p)"]); A["FM t"] = A.col.map(fmres["t"])
A["방향 일치"] = np.where(np.sign(A["SHAP 방향(순위~SHAP 상관)"]) == np.sign(A["FM 계수(%p)"]), "같음", "다름")
A = A.sort_values("중요도(평균|SHAP|)", ascending=False)
A.to_csv(f"{OUT}/shap_vs_fm.csv", index=False, encoding="utf-8-sig")
print(A.round(4).to_string(index=False))

# 그림: 중요도 막대 + 10분위 dependence
dec = pd.DataFrame({c: S[c + "_r"].groupby(pd.cut(va[c + "_r"], np.linspace(0, 1, 11), include_lowest=True), observed=False).mean().values for c in ALL})
fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharey=True)
for a, c in zip(axes.flat, A.col):
    v = dec[c].values * 100
    a.bar(range(1, 11), v, color=["#1F5C5A" if x >= 0 else "#B4532A" for x in np.nan_to_num(v)], width=0.7)
    a.axhline(0, color="#8C877D", lw=0.8)
    a.set_title(f"{NM[c]}  (중요도 {A.set_index('col').loc[c,'중요도(평균|SHAP|)']*100:.2f})", loc="left", fontsize=15)
    a.set_xticks([1, 10]); a.set_xticklabels(["값 낮음", "값 높음"])
for a in axes[:, 0]: a.set_ylabel("예측 순위에 더한 값\n(SHAP, ×100)")
fig.suptitle("XGBoost가 본 방향: 변수 10분위별 평균 SHAP (2011–2021 검증 표본 2만 개)", fontsize=17)
fig.tight_layout(); fig.savefig(f"{OUT}/figK_shap_dependence.png", dpi=140)

# ---------- B. FM 예측 vs XGBoost 예측 ----------
u = df[df.target.notna() & (df.month >= "2002-04")]
B = []
for m, g in u.groupby("month"):
    X = np.column_stack([np.ones(len(g))] + [g[c + "_r"].values for c in ALL])
    b = np.linalg.lstsq(X, g.target.values, rcond=None)[0]
    B.append([m] + list(b))
B = pd.DataFrame(B, columns=["month", "a"] + ALL).set_index("month")
pred = []
for Y in range(2011, 2022):
    coef = B[B.index <= f"{Y-1}-11"][ALL].mean()                 # 그때까지 알 수 있는 계수 평균
    te = df[(df.month >= f"{Y}-01") & (df.month <= f"{Y}-12") & df.target.notna()].copy()
    te["fm_score"] = te[[c + "_r" for c in ALL]].values @ coef.values
    pred.append(te[["month", "code", "target", "fm_score"]])
P = pd.concat(pred)
X3 = pd.read_parquet(f"{OUT}/pred_3.parquet")[["month", "code", "score"]].rename(columns={"score": "xgb_score"})
P = P.merge(X3, on=["month", "code"], how="inner")
def mic(col): return P.groupby("month").apply(lambda x: x[col].rank().corr(x.target.rank()), include_groups=False)
ic_fm, ic_x = mic("fm_score"), mic("xgb_score")
agree = P.groupby("month").apply(lambda x: x.fm_score.rank().corr(x.xgb_score.rank()), include_groups=False)
d = ic_x - ic_fm
res = {"FM 예측 평균 IC": ic_fm.mean(), "FM t": ic_fm.mean() / (ic_fm.std() / np.sqrt(len(ic_fm))),
       "XGB 예측 평균 IC": ic_x.mean(), "XGB t": ic_x.mean() / (ic_x.std() / np.sqrt(len(ic_x))),
       "차이(XGB−FM)": d.mean(), "차이 t": d.mean() / (d.std() / np.sqrt(len(d))),
       "XGB가 이긴 해": int((ic_x.groupby(ic_x.index.str[:4]).mean() > ic_fm.groupby(ic_fm.index.str[:4]).mean()).sum()),
       "두 예측의 순위상관(월평균)": agree.mean()}
pd.Series(res).to_csv(f"{OUT}/fm_vs_xgb_pred.csv", encoding="utf-8-sig")
print({k: round(v, 4) for k, v in res.items()})
yr = pd.DataFrame({"FM": ic_fm.groupby(ic_fm.index.str[:4]).mean(), "XGBoost": ic_x.groupby(ic_x.index.str[:4]).mean()})
yr.to_csv(f"{OUT}/fm_vs_xgb_by_year.csv", encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(13, 5)); x = np.arange(len(yr))
ax.bar(x - 0.2, yr.FM, 0.4, color="#C9C3B6", label=f"Fama-MacBeth 예측 (평균 {res['FM 예측 평균 IC']:.3f})")
ax.bar(x + 0.2, yr.XGBoost, 0.4, color="#1F5C5A", label=f"XGBoost 예측 (평균 {res['XGB 예측 평균 IC']:.3f})")
ax.set_xticks(x); ax.set_xticklabels(yr.index); ax.set_ylabel("그해 평균 IC"); ax.legend(frameon=False)
ax.set_title(f"같은 9개 변수, 같은 누적 방식: 두 모델의 예측력 (두 예측의 순위상관 {res['두 예측의 순위상관(월평균)']:.2f})", loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/figL_fm_vs_xgb.png", dpi=140)
