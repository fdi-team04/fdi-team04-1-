"""9장 그림 f9_sent_full.png: 왼쪽 감성 IC(같은 달/다음 달/2달 뒤), 오른쪽 기간별 FM 계수 (sent_ic.csv, sent_fm_by_period.csv 사용)"""
import pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, koreanize_matplotlib  # noqa
P = "/home/user/fdi-team04-1-/연습장/out_yw/news_sent_full"
fm = pd.read_csv(f"{P}/sent_fm_by_period.csv"); ic = pd.read_csv(f"{P}/sent_ic.csv")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.3]})
r = ic.iloc[0]; vals = [r["같은달 IC"], r["다음달 IC"], r["2달뒤 IC"]]; ts = [r["t(같은달)"], r["t(다음달)"], r["t(2달뒤)"]]
b = ax[0].bar(["같은 달", "다음 달", "2달 뒤"], vals, color=["#1E4E8C", "#5B8FD6", "#A9C4EA"])
for bb, v, t in zip(b, vals, ts): ax[0].text(bb.get_x() + bb.get_width() / 2, v + 0.004, f"{v:.3f}\n(t {t:.1f})", ha="center", fontsize=10)
ax[0].set_ylim(0, 0.22); ax[0].set_title("감성 점수와 수익률 순위의 상관 (IC, 2001–2021)", fontsize=11); ax[0].spines[["top", "right"]].set_visible(False)
s = fm[fm.변수 == "s_all"].set_index("기간").loc[["2002-10", "2011-15", "2016-21"]]
cols = ["#1E4E8C" if abs(t) >= 2 else "#B8C4D6" for t in s.t]
b = ax[1].bar(["2002–2010", "2011–2015", "2016–2021"], s.계수, color=cols); ax[1].axhline(0, color="#555", lw=0.8)
for bb, v, t in zip(b, s.계수, s.t): ax[1].text(bb.get_x() + bb.get_width() / 2, v + (0.05 if v >= 0 else -0.15), f"{v:+.2f}%p (t {t:.1f})", ha="center", fontsize=10)
ax[1].set_ylim(-0.5, 1.4); ax[1].set_ylabel("다음 달 수익률 차이 (%p/월)")
ax[1].set_title("기간별 FM 계수: 다른 변수 9개를 통제한 감성 효과", fontsize=11); ax[1].spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(f"{P}/f9_sent_full.png", dpi=200)
