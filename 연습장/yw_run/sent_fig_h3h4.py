"""H3·H4 판정 그림 (전체 감성 129만 건, 4장 figM과 같은 모양: 기간별 막대, * = |t|≥2)"""
import numpy as np, pandas as pd, matplotlib
src = open("/home/user/fdi-team04-1-/연습장/yw_run/sent_analysis_yw.py").read().split("# FM: 통제")[0]
exec(src)
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
FD = "/tmp/claude-0/-home-user-fdi-team04-1-/03b15712-f400-5188-9b23-fea64cb12856/scratchpad/fonts"
for f in ("plex400.ttf", "plex600.ttf"): fm.fontManager.addfont(f"{FD}/{f}")
plt.rcParams.update({"font.family": "IBM Plex Sans KR", "axes.unicode_minus": False, "font.size": 13,
                     "axes.spines.top": False, "axes.spines.right": False})
PER = [("2001–2010", "2001-01", "2010-12"), ("2011–2015", "2011-01", "2015-12"), ("2016–2021", "2016-01", "2021-12")]
def by_per(d):
    return {p: (d[(d.index >= a) & (d.index <= b)].mean(), tt(d[(d.index >= a) & (d.index <= b)])) for p, a, b in PER}
R = {}
# H3: 논조 상위 20% − 하위 20%, 다음 달
for col, lab in [("s_all", "전체 논조"), ("s_noprice", "주가 보도 뺀 논조"), ("surprise", "논조 변화\n(평소 대비)")]:
    x = u[u[col].notna()].copy()
    x["q"] = x.groupby("month")[col].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
    g = x.groupby(["month", "q"]).target.mean().unstack() * 100
    R["H3 " + lab] = by_per((g[5] - g[1]).dropna())
# H4: 호재(논조 상위 20%) 종목에서 개인 순매수 많은 절반 − 적은 절반, 그리고 나머지 종목 대비 차이의 차이
x = u[u.s_all.notna() & u.retail_nb.notna()].copy()
x["good"] = x.groupby("month").s_all.rank(pct=True) > 0.8
x["rb"] = x.groupby(["month", "good"]).retail_nb.rank(pct=True) > 0.5
g = x.groupby(["month", "good", "rb"]).target.mean().unstack(["good", "rb"]) * 100
dg = (g[(True, True)] - g[(True, False)]).dropna(); do = (g[(False, True)] - g[(False, False)]).dropna()
R["H4 호재 종목:\n개인 많이−적게"] = by_per(dg); R["H4 나머지 종목:\n개인 많이−적게"] = by_per(do); R["H4 차이의 차이"] = by_per((dg - do).dropna())
rows = [{"항목": k.replace("\n", " "), "기간": p, "차이(%p)": v[0], "t": v[1]} for k, d in R.items() for p, v in d.items()]
pd.DataFrame(rows).to_csv(f"{P}/h3h4_by_period.csv", index=False, encoding="utf-8-sig")
print(pd.DataFrame(rows).round(3).to_string(index=False))

labs = list(R); x0 = np.arange(len(labs)) + np.array([0, 0, 0, 0.6, 0.6, 0.6])
fig, ax = plt.subplots(figsize=(13, 6.2)); cols = ["#C5CED9", "#5B8FCF", "#1E4E8C"]
for i, (p, _, _) in enumerate(PER):
    v = [R[k][p][0] for k in labs]; t = [R[k][p][1] for k in labs]
    ax.bar(x0 + (i - 1) * 0.27, v, 0.27, color=cols[i], label=p)
    for j in range(len(labs)):
        if abs(t[j]) >= 2: ax.text(x0[j] + (i - 1) * 0.27, v[j] + (0.05 if v[j] >= 0 else -0.14), "*", ha="center", fontsize=14)
ax.axhline(0, color="#8C877D", lw=0.8); ax.axvline(2.8, color="#B9B4AA", lw=0.8, ls="--")
ax.set_xticks(x0); ax.set_xticklabels([k.split(" ", 1)[1] for k in labs], fontsize=12)
ax.set_ylim(-1.0, 1.75)
ax.text(1, 1.62, "H3  논조 상위 20% − 하위 20%", ha="center", fontsize=13, color="#1E4E8C", weight=600)
ax.text(4.6, 1.62, "H4  호재 종목에서 개인 순매수 많이 − 적게", ha="center", fontsize=13, color="#1E4E8C", weight=600)
ax.set_ylabel("다음 달 수익률 차 (%p/월)"); ax.legend(frameon=False, loc="lower left")
ax.set_title("감성 점수로 본 H3·H4, 기간별 (전체 129만 건, * = |t|≥2)", loc="left")
fig.tight_layout(); fig.savefig(f"{P}/figH3H4_by_period.png", dpi=140)
fig.savefig("/home/user/fdi-team04-1-/연습장/해설서/img_final/figH3H4_by_period.png", dpi=140)
