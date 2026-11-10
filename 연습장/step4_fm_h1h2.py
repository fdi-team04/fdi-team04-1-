"""3단계: 보조 판정 — Fama-MacBeth(방향) + H1·H2 수급 5분위 표

Fama-MacBeth: 매달 단면 OLS  target(%) = a + Σ b_k · rank_k(0~1)  → b_k 월평균, t = 평균/(표준편차/√n)
  b_k 해석: 다른 변수가 같을 때, 변수 k가 그달 꼴찌→1등으로 갈 때 다음 달 수익률 차이(%p)
  판정: 부호가 가설 방향 & |t|≥2 "어느 정도 지지", |t|≥3 "확실하게 지지", 그 외 "확인하지 못함"
  (Newey-West 3개월 보정 t도 같이 적음: 월별 계수의 자기상관 대비)
H1·H2 수급 표: 순매수 비율 상위20%(Q5) − 하위20%(Q1) 다음 달 수익률 차이, 월평균과 t
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, koreanize_matplotlib  # noqa
plt.rcParams["axes.unicode_minus"] = False
OUT = "/home/user/fdi-team04-1-/연습장/out"
ALL = ["beta", "size", "bm", "rev", "mom", "ivol", "issue", "retail_nb", "foreign_nb"]
NAME = {"beta": "베타", "size": "규모", "bm": "가치", "rev": "단기 반전", "mom": "모멘텀", "ivol": "고유변동성",
        "issue": "주식 발행", "retail_nb": "개인 순매수(H1)", "foreign_nb": "외국인 순매수(H2)"}
EXPECT = {"beta": "+", "size": "-", "bm": "+", "rev": "-", "mom": "±", "ivol": "-", "issue": "-",
          "retail_nb": "-", "foreign_nb": "+"}

df = pd.read_parquet(f"{OUT}/features.parquet")
df = df[(df.month >= "2002-04") & df.target.notna()].copy()     # 가치 변수가 시작되는 달부터
for c in ALL:
    df[c + "_r"] = df.groupby("month")[c].rank(pct=True).fillna(0.5)

def nw_t(x, lag=3):
    x = np.asarray(x) - np.mean(x); n = len(x); g0 = x @ x / n; s = g0
    for l in range(1, lag + 1):
        s += 2 * (1 - l / (lag + 1)) * (x[l:] @ x[:-l] / n)
    return None, np.sqrt(s / n)

def fm(cols, start, end):
    coefs = []
    for m, g in df[(df.month >= start) & (df.month <= end)].groupby("month"):
        X = np.column_stack([np.ones(len(g))] + [g[c + "_r"].values for c in cols])
        b, *_ = np.linalg.lstsq(X, g.target.values * 100, rcond=None)
        coefs.append(b[1:])
    B = pd.DataFrame(coefs, columns=cols)
    out = []
    for c in cols:
        x = B[c]; t = x.mean() / (x.std() / np.sqrt(len(x)))
        _, se_nw = nw_t(x.values); tnw = x.mean() / se_nw
        exp = EXPECT[c]; ok_dir = (exp == "±") or (np.sign(x.mean()) == (1 if exp == "+" else -1))
        if exp == "±": verdict = "방향 가설 없음(통제)"
        elif ok_dir and abs(t) >= 3: verdict = "확실하게 지지"
        elif ok_dir and abs(t) >= 2: verdict = "어느 정도 지지"
        elif (not ok_dir) and abs(t) >= 2: verdict = "반대 방향으로 유의"
        else: verdict = "확인하지 못함"
        out.append({"변수": NAME[c], "col": c, "예상": exp, "계수(%p)": x.mean(), "t": t, "t(NW3)": tnw,
                    "판정": verdict, "달수": len(x)})
    return pd.DataFrame(out)

res = []
for label, s, e in [("검증 2011~2021", "2011-01", "2021-11"), ("전체 2002~2021", "2002-04", "2021-11")]:
    r = fm(ALL, s, e).assign(구간=label); res.append(r)
    print(label); print(r.round(3).to_string(index=False))
# 수급만 기준선+통제와 같이 넣은 버전(보강 변수 없이) — 결과가 보강 변수 포함 여부에 민감한지
r = fm(["beta", "size", "bm", "rev", "mom", "retail_nb", "foreign_nb"], "2011-01", "2021-11").assign(구간="검증, 보강변수 제외")
res.append(r); print(r.round(3).to_string(index=False))
# 수급 단독
r = fm(["retail_nb", "foreign_nb"], "2011-01", "2021-11").assign(구간="검증, 수급 단독"); res.append(r)
print(r.round(3).to_string(index=False))
FM = pd.concat(res); FM.to_csv(f"{OUT}/fama_macbeth.csv", index=False, encoding="utf-8-sig")

# ---- H1·H2 수급 5분위 표 (2011~2021) + 대형/소형 분리
def qtable(d, v):
    d = d[["month", v, "target", "me"]].dropna().copy()
    d["q"] = d.groupby("month")[v].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
    qm = d.groupby(["month", "q"]).target.mean().unstack() * 100
    sp = qm[5] - qm[1]
    return qm.mean(), sp.mean(), sp.mean() / (sp.std() / np.sqrt(len(sp))), sp
val = df[df.month >= "2011-01"].copy()
val["big"] = val.groupby("month").me.transform(lambda s: s.rank(pct=True) > 0.5)
rows, sps = [], {}
for v, h, exp in [("retail_nb", "H1", "-"), ("foreign_nb", "H2", "+")]:
    for grp, d in [("전체", val), ("대형(시총 상위 50%)", val[val.big]), ("소형(하위 50%)", val[~val.big])]:
        qm, sp, t, s = qtable(d, v); sps[(v, grp)] = s
        ok = np.sign(sp) == (1 if exp == "+" else -1)
        verdict = ("확실하게 지지" if ok and abs(t) >= 3 else "어느 정도 지지" if ok and abs(t) >= 2
                   else "반대 방향으로 유의" if (not ok) and abs(t) >= 2 else "확인하지 못함")
        rows.append({"가설": h, "변수": NAME[v], "그룹": grp, **{f"Q{q}": qm[q] for q in range(1, 6)},
                     "Q5-Q1(%p)": sp, "t": t, "판정": verdict})
QT = pd.DataFrame(rows); QT.to_csv(f"{OUT}/h1h2_quintile.csv", index=False, encoding="utf-8-sig")
print(QT.round(3).to_string(index=False))

# 순매수와 단기 반전의 관계 (같은 달 상관) — 해석용
cor = val.groupby("month").apply(lambda x: pd.Series({
    "개인순매수~t월수익률": x.retail_nb.rank().corr(x.ret.rank()),
    "외국인순매수~t월수익률": x.foreign_nb.rank().corr(x.ret.rank()),
    "개인~외국인": x.retail_nb.rank().corr(x.foreign_nb.rank())}), include_groups=False).mean()
cor.to_csv(f"{OUT}/nb_correlations.csv", encoding="utf-8-sig"); print(cor.round(3))

# ---- 그림 E: 순매수 월별 추이 (선그래프) — 2001~2021, 시총 가중 평균과 중앙값
raw = pd.read_parquet(f"{OUT}/features.parquet"); raw = raw[raw.month >= "2001-01"]
agg = raw.groupby("month").apply(lambda x: pd.Series({
    "개인_중앙값": x.retail_nb.median() * 100, "외국인_중앙값": x.foreign_nb.median() * 100,
    "개인_순매수종목비율": (x.retail_nb > 0).mean() * 100, "외국인_순매수종목비율": (x.foreign_nb > 0).mean() * 100}),
    include_groups=False)
agg.to_csv(f"{OUT}/nb_monthly.csv", encoding="utf-8-sig")
fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
t = pd.to_datetime(agg.index)
ax[0].plot(t, agg["개인_중앙값"].rolling(3).mean(), label="개인", color="#c44e52")
ax[0].plot(t, agg["외국인_중앙값"].rolling(3).mean(), label="외국인", color="#4c72b0")
ax[0].axhline(0, color="k", lw=.5); ax[0].set_ylabel("순매수 ÷ 전월 시총 (%)\n중앙값, 3개월 평균"); ax[0].legend()
ax[0].set_title("그림 E. 개인·외국인 순매수 비율의 월별 흐름 (2001~2021)")
ax[1].plot(t, agg["개인_순매수종목비율"].rolling(3).mean(), color="#c44e52", label="개인이 순매수한 종목 비율")
ax[1].plot(t, agg["외국인_순매수종목비율"].rolling(3).mean(), color="#4c72b0", label="외국인이 순매수한 종목 비율")
ax[1].axhline(50, color="k", lw=.5); ax[1].set_ylabel("%"); ax[1].legend()
fig.tight_layout(); fig.savefig(f"{OUT}/figE_nb_timeseries.png", dpi=130)

# ---- 그림 F: H1·H2 5분위 막대 (전체/대형/소형)
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=True)
for i, v in enumerate(["retail_nb", "foreign_nb"]):
    for k, grp in enumerate(["전체", "대형(시총 상위 50%)", "소형(하위 50%)"]):
        r = QT[(QT.변수 == NAME[v]) & (QT.그룹 == grp)].iloc[0]
        a = axes[i, k]; vals = [r[f"Q{q}"] for q in range(1, 6)]
        a.bar(range(1, 6), vals, color=["#c44e52" if q in (1, 5) else "#8c8c8c" for q in range(1, 6)])
        a.set_title(f"{NAME[v]} · {grp}\nQ5-Q1 {r['Q5-Q1(%p)']:+.2f}%p (t {r['t']:+.1f}) → {r['판정']}", fontsize=10)
        a.set_xticks(range(1, 6)); a.set_xticklabels(["1\n많이 팖", "2", "3", "4", "5\n많이 삼"])
        if k == 0: a.set_ylabel("다음 달 평균 수익률(%)")
fig.suptitle("그림 F. H1·H2 수급 표: 순매수 비율 5분위별 다음 달 수익률 (2011~2021)", fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/figF_h1h2_quintiles.png", dpi=130)

# ---- 그림 G: Fama-MacBeth 계수와 t (검증 구간)
r = FM[FM.구간 == "검증 2011~2021"]
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#55a868" if "지지" in v else "#c44e52" if "반대" in v else "#8c8c8c" for v in r.판정]
ax.barh(r.변수[::-1], r["t"][::-1], color=colors[::-1])
for x in (-3, -2, 2, 3): ax.axvline(x, color="k", ls=":" if abs(x) == 2 else "--", lw=.8)
for y, (tv, cf, ex) in enumerate(zip(r["t"][::-1], r["계수(%p)"][::-1], r["예상"][::-1])):
    ax.text(tv + (0.15 if tv >= 0 else -0.15), y, f"계수 {cf:+.2f}%p · 예상 {ex}", va="center",
            ha="left" if tv >= 0 else "right", fontsize=9)
ax.set_xlabel("t값 (점선 ±2: 어느 정도, 파선 ±3: 확실)")
ax.set_title("그림 G. Fama-MacBeth 방향 확인 (9개 변수 동시, 2011~2021)\n초록=가설 방향 지지, 빨강=반대 방향 유의, 회색=확인 못함")
ax.set_xlim(-9, 9); fig.tight_layout(); fig.savefig(f"{OUT}/figG_fama_macbeth.png", dpi=130)
