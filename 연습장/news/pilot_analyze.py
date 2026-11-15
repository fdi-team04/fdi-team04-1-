"""뉴스 파일럿 2: 감성 점수가 수익률과 관계가 있나 (이안 제안: "긍정확률−부정확률이 얼마나 유의미한 수치로 작동하는지 파일럿")

입력: out/news_pilot/pilot_news_monthly.csv (pilot_build.py), out/features.parquet
표본: 파일럿에서 기사가 1건 이상 있는 (종목, 연도)만 '수집됨'으로 본다 → 그 안에서만 비교
      (수집 안 된 종목을 '기사 없음'으로 착각하지 않도록. 본 분석 step9와 같은 원칙)
보는 것 (왜):
 0. 점수 타당성: 같은 달 수익률과의 상관 — 좋은 기사가 나온 달에 주가가 올랐다면 점수가 뭔가를 잡는다는 뜻
 1. 다음 달 예측: 매월 횡단면 순위상관(IC) — tone(m) vs 다음 달 수익률. 전체 논조 / 화제형 논조 / 공시형 / 논란형 / 기사 수
 2. 논조 3분위 다음 달 수익률 (상−하)
 3. Fama-MacBeth: 9개 통제 + 화제 논조 → 다른 변수로 설명되지 않는 부분이 남나
 4. H4 설거지 표 (파일럿이라 표본 작음 → 방향만 참고)
 5. 기간별(2001–10 / 2011–21) 안정성
판정은 하지 않는다. 파일럿 = 본 데이터(아영 정리본 전체)로 돌릴 때 무엇을 볼지 정하는 예비 확인.
"""
import numpy as np, pandas as pd, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
FD = "/tmp/claude-0/-home-user-fdi-team04-1-/03b15712-f400-5188-9b23-fea64cb12856/scratchpad/fonts"
for f in ("plex400.ttf", "plex600.ttf"): fm.fontManager.addfont(f"{FD}/{f}")
plt.rcParams.update({"font.family": "IBM Plex Sans KR", "axes.unicode_minus": False, "font.size": 13,
                     "axes.spines.top": False, "axes.spines.right": False})
O = "/home/user/fdi-team04-1-/연습장/out"; P = f"{O}/news_pilot"
ALL = ["beta", "size", "bm", "rev", "mom", "ivol", "issue", "retail_nb", "foreign_nb"]
def tt(x): x = pd.Series(x).dropna(); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 else np.nan
R = {}

f = pd.read_parquet(f"{O}/features.parquet")
nm = pd.read_csv(f"{P}/pilot_news_monthly.csv", dtype={"code": str})
cy = nm.assign(year=nm.month.str[:4])[["code", "year"]].drop_duplicates()
f["year"] = f.month.str[:4]
u = f.merge(cy, on=["code", "year"]).merge(nm, on=["month", "code"], how="left")
for c in [c for c in nm if c.startswith("n_")]: u[c] = u[c].fillna(0)
u = u[u.target.notna()]
u["ret_r"] = u.groupby("month").ret.rank(pct=True); u["y"] = u.groupby("month").target.rank(pct=True)
u["log_n"] = np.log1p(u.n_all)
R["표본 종목-월"] = len(u); R["종목"] = u.code.nunique(); R["달"] = u.month.nunique()
R["기사 있는 종목-월 비율"] = round((u.n_all > 0).mean(), 3); R["화제형 있는 비율"] = round((u.n_topic > 0).mean(), 3)

def mic(x, col, ycol, minn=8):
    g = x[x[col].notna()].groupby("month")
    s = g.apply(lambda z: z[col].rank().corr(z[ycol].rank()) if len(z) >= minn else np.nan, include_groups=False).dropna()
    return s
rows = []
for col, nmk in [("all_tone", "전체 논조"), ("topic_tone", "화제형 논조"), ("disc_tone", "공시형 논조"),
                 ("contro_tone", "논란형 논조"), ("etc_tone", "기타 논조"), ("log_n", "기사 수(log)")]:
    for lab, a, b in [("전체", "2001-01", "2021-12"), ("2001–10", "2001-01", "2010-12"), ("2011–21", "2011-01", "2021-12")]:
        x = u[(u.month >= a) & (u.month <= b)]
        c0, c1 = mic(x, col, "ret_r"), mic(x, col, "y")
        rows.append({"변수": nmk, "기간": lab, "같은달 IC": c0.mean(), "t(같은달)": tt(c0),
                     "다음달 IC": c1.mean(), "t(다음달)": tt(c1), "달수": len(c1),
                     "칸 수": int(x[col].notna().sum())})
IC = pd.DataFrame(rows); IC.to_csv(f"{P}/pilot_ic.csv", index=False, encoding="utf-8-sig")
print(IC.round(3).to_string(index=False))

# 2. 논조 3분위 (전체 논조, 기사 있는 종목만) — 같은 달 / 다음 달
w = u[u.all_tone.notna()].copy()
w["q"] = w.groupby("month").all_tone.transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False) + 1 if len(s) >= 9 else np.nan)
T3 = {}
for ycol, lab in [("ret", "같은 달"), ("target", "다음 달")]:
    qm = w.groupby(["month", "q"])[ycol].mean().unstack() * 100
    T3[lab] = {"하(부정)": qm[1].mean(), "중": qm[2].mean(), "상(긍정)": qm[3].mean(), "상−하": (qm[3] - qm[1]).mean(), "t": tt(qm[3] - qm[1]), "달수": int((qm[3]-qm[1]).notna().sum())}
T3 = pd.DataFrame(T3).T; T3.to_csv(f"{P}/pilot_tercile.csv", encoding="utf-8-sig"); print(T3.round(3))

# 3. Fama-MacBeth: 통제 9 + 논조 (+ 기사 없음 표시)
for c in ALL + ["all_tone", "topic_tone"]:
    u[c + "_r"] = u.groupby("month")[c].rank(pct=True).fillna(0.5)
u["no_news"] = (u.n_all == 0).astype(float); u["no_topic"] = (u.n_topic == 0).astype(float)
def fmreg(cols, x):
    B = []
    for m, g in x.groupby("month"):
        if len(g) < len(cols) + 5: continue
        X = np.column_stack([np.ones(len(g))] + [g[c].values for c in cols])
        B.append(np.linalg.lstsq(X, g.target.values * 100, rcond=None)[0][1:])
    B = pd.DataFrame(B, columns=cols)
    return pd.DataFrame({"계수(%p)": B.mean(), "t": B.apply(tt), "달수": len(B)})
FMr = pd.concat({"통제+전체논조": fmreg([c + "_r" for c in ALL] + ["all_tone_r", "no_news"], u),
                 "통제+화제논조": fmreg([c + "_r" for c in ALL] + ["topic_tone_r", "no_topic"], u)}, axis=1)
FMr.to_csv(f"{P}/pilot_fm.csv", encoding="utf-8-sig"); print(FMr.round(3).to_string())

# 4. H4 (전체 논조로: 화제형만으로는 칸이 너무 적음)
res = []
for m, g in u.groupby("month"):
    news = g[g.all_tone.notna()]
    if len(news) < 8: continue
    good = news[news.all_tone > news.all_tone.median()]; nonews = g[g.n_all == 0]
    rmed = g.retail_nb.median()
    def diff(x):
        hi, lo = x[x.retail_nb > rmed], x[x.retail_nb <= rmed]
        return (hi.target.mean() - lo.target.mean()) * 100 if len(hi) >= 2 and len(lo) >= 2 else np.nan
    res.append({"month": m, "호재차이": diff(good), "기사없음차이": diff(nonews), "논조하위차이": diff(news[news.all_tone <= news.all_tone.median()])})
H = pd.DataFrame(res).set_index("month"); H["DiD(호재−기사없음)"] = H.호재차이 - H.기사없음차이; H["DiD(호재−악재)"] = H.호재차이 - H.논조하위차이
H4 = pd.DataFrame({"평균(%p)": H.mean(), "t": H.apply(tt), "달수": H.notna().sum()})
H4.to_csv(f"{P}/pilot_h4.csv", encoding="utf-8-sig"); print(H4.round(3))

# 그림: 같은달/다음달 IC 막대
x = IC[IC.기간 == "전체"].set_index("변수")
fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
for a, col, tcol, ttl in [(ax[0], "같은달 IC", "t(같은달)", "같은 달 수익률과 (점수가 뉴스를 제대로 읽나)"), (ax[1], "다음달 IC", "t(다음달)", "다음 달 수익률과 (예측력)")]:
    v = x[col]; a.barh(range(len(v)), v, color=["#1F5C5A" if t >= 2 else "#B4532A" if t <= -2 else "#C9C3B6" for t in x[tcol]])
    a.set_yticks(range(len(v))); a.set_yticklabels(v.index); a.axvline(0, color="#8C877D", lw=0.8); a.invert_yaxis()
    for i, (vv, t) in enumerate(zip(v, x[tcol])): a.text(vv if vv >= 0 else 0, i, f"  t {t:.1f}", va="center", ha="left", fontsize=11)
    a.set_title(ttl, loc="left", fontsize=14); a.set_xlabel("월평균 순위상관")
fig.suptitle("파일럿: KR-FinBert-SC 논조(긍정확률−부정확률)와 수익률 (색 = |t|≥2)", fontsize=16)
fig.tight_layout(); fig.savefig(f"{P}/figP_pilot_ic.png", dpi=140)
json.dump(R, open(f"{P}/pilot_sample.json", "w"), ensure_ascii=False, indent=1); print(R)
