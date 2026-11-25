import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; from matplotlib import font_manager as fm
fm.fontManager.addfont("/tmp/claude-0/-home-user-fdi-team04-1-/03b15712-f400-5188-9b23-fea64cb12856/scratchpad/fonts/gowun.ttf")
plt.rcParams.update({"font.family": "Gowun Dodum", "axes.unicode_minus": False, "font.size": 11})
O = "/home/user/fdi-team04-1-/연습장/out_yw"; D = "/home/user/fdi-team04-1-/연습장/기록/feat_fig"
KO = {"mom6": "6개월 모멘텀", "ltrev": "장기 반전", "high52": "52주 고점 근접", "maxret": "최대 월수익률", "skew": "수익률 왜도", "season": "같은 달 계절성",
      "consist": "상승한 달 비율", "price": "주가 수준", "idvol": "고유변동성", "retail3": "개인 3개월 순매수", "foreign3": "외국인 3개월 순매수",
      "bgrowth": "장부가치 성장", "beta_cd (베타×금리변화)": "금리 변화(×베타)"}
s = pd.read_csv(f"{O}/feat_screen.csv"); s["이름"] = s["피처"].map(KO); s = s.sort_values("IC 차이")
fig, ax = plt.subplots(figsize=(8, 5.2), dpi=200)
col = ["#257de0" if b else "#c9cdd2" for b in s.B]
ax.barh(s["이름"], s["IC 차이"] * 1000, color=col, height=0.62)
ax.axvline(0, color="#666666", lw=0.8)
for i, (v, b, n) in enumerate(zip(s["IC 차이"] * 1000, s.B, s["오른 해"])):
    ax.text(v + (0.08 if v >= 0 else -0.08), i, f"{v:+.1f}  ({n}/11년)", va="center", ha="left" if v >= 0 else "right", fontsize=9.5, color="#191919")
ax.set_xlabel("XGBoost 평균 IC 변화 (×0.001)  ·  현재 모델(정형 9개 + 목표가) 대비", color="#666666")
ax.set_xlim(-3, 5.2); ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(axis="y", length=0); ax.grid(axis="x", color="#eeeeee"); ax.set_axisbelow(True)
ax.set_title("파란색 = B 단계 통과 (평균 IC 상승 + 11년 중 6년 이상)", loc="left", fontsize=10, color="#666666")
plt.tight_layout(); plt.savefig(f"{D}/stageB.png"); plt.close()
b = pd.read_csv(f"{O}/feat_bundle.csv"); models = ["XGBoost", "LightGBM", "랜덤포레스트", "신경망", "릿지"]
base = b.groupby("모델")["기준 IC"].first().reindex(models); m2 = b[b["묶음"] == "+mom6+idvol"].set_index("모델").IC.reindex(models); m7 = b[b["묶음"] == "+B통과 7개"].set_index("모델").IC.reindex(models)
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200); x = np.arange(len(models))
for xi in x: ax.plot([xi, xi], [min(base.iloc[xi], m2.iloc[xi], m7.iloc[xi]), max(base.iloc[xi], m2.iloc[xi], m7.iloc[xi])], color="#dddddd", lw=2, zorder=1)
for v, lab, c, mk in [(base, "현재 모델", "#257de0", "o"), (m2, "+ 고유변동성·6개월 모멘텀", "#f47230", "s"), (m7, "+ B 통과 7개 전부", "#00b894", "D")]:
    ax.scatter(x, v, s=70, label=lab, color=c, marker=mk, zorder=3, edgecolor="white", linewidth=1.5)
ax.set_xticks(x, models); ax.set_ylabel("평균 IC (2011–2021)", color="#666666")
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#eeeeee"); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=9.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
plt.tight_layout(); plt.savefig(f"{D}/bundle.png"); plt.close(); print("ok")
