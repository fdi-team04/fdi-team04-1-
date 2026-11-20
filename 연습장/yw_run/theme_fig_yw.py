"""테마 판정 그림: 테마×(긍정/부정) 다음 달 수익률 차이를 기간 3개로 (4장 figM과 같은 모양, * = |t|>=2)"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
FD = "/tmp/claude-0/-home-user-fdi-team04-1-/03b15712-f400-5188-9b23-fea64cb12856/scratchpad/fonts"
for f_ in ("plex400.ttf", "plex600.ttf"): fm.fontManager.addfont(f"{FD}/{f_}")
plt.rcParams.update({"font.family": "IBM Plex Sans KR", "axes.unicode_minus": False, "font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
P = "/home/user/fdi-team04-1-/연습장/out_yw/theme_judge_full"; T = pd.read_csv(f"{P}/theme_judge.csv")
items = [(r.테마, s) for _, r in T.iterrows() for s in ["긍정", "부정"] if not pd.isna(r.get(f"{s} 2001–10", np.nan))]
PER = ["2001–10", "2011–15", "2016–21"]; cols = ["#C5CED9", "#5B8FCF", "#1E4E8C"]
fig, ax = plt.subplots(figsize=(13, 5.4)); x = np.arange(len(items))
for i, p in enumerate(PER):
    v = [T.set_index("테마").loc[g, f"{s} {p}"] for g, s in items]; t = [T.set_index("테마").loc[g, f"{s} {p} t"] for g, s in items]
    ax.bar(x + (i - 1) * 0.27, v, 0.27, color=cols[i], label=p)
    for j in range(len(items)):
        if abs(t[j]) >= 2: ax.text(x[j] + (i - 1) * 0.27, v[j] + (0.06 if v[j] >= 0 else -0.2), "*", ha="center", fontsize=15)
ax.axhline(0, color="#8C877D", lw=0.8); ax.set_xticks(x); ax.set_xticklabels([f"{g}\n{s}" for g, s in items], fontsize=10.5)
ax.set_ylabel("다음 달 수익률 차이 (%p/월)\n(테마 기사 있는 종목 − 없는 종목)"); ax.set_ylim(-1.15, 1.75); ax.legend(frameon=False, ncol=3, loc="upper left")
ax.set_title("뉴스 테마별 다음 달 효과, 기간별 (상위 200, * = |t|≥2)", loc="left")
fig.tight_layout(); fig.savefig(f"{P}/fig_theme_by_period.png", dpi=140)
