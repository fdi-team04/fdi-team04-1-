"""전체 감성 보강: (1) ESG 부정 있음−없음·ESG 논조를 기간별로 (2) 원래 H4: 호재(논조 상위)+개인 순매수 → 다음 달 약한가 (기간별)"""
import numpy as np, pandas as pd, json
src = open("/home/user/fdi-team04-1-/연습장/yw_run/sent_analysis_yw.py").read().split("# FM: 통제")[0]
exec(src)
PER = [("2001–10", "2001-01", "2010-12"), ("2011–15", "2011-01", "2015-12"), ("2016–21", "2016-01", "2021-12"), ("전체", "2001-01", "2021-12")]
out = {}
# (1) ESG 부정 있음-없음 다음 달, 기간별
u["neg"] = (u.n_esg_neg.fillna(0) > 0).astype(float); u["pos"] = (u.n_esg_pos.fillna(0) > 0).astype(float)
rows = []
for flag, lab in [("neg", "부정 ESG 있음−없음"), ("pos", "긍정 ESG 있음−없음")]:
    d = (u.groupby(["month", flag]).target.mean().unstack() * 100); d = (d[1] - d[0]).dropna()
    for nm_, a, b in PER:
        z = d[(d.index >= a) & (d.index <= b)]; rows.append({"비교": lab, "기간": nm_, "다음달 차이(%p)": z.mean(), "t": tt(z)})
# ESG 부정 기사 논조: 부정 기사 있는 종목끼리 논조 순위 vs 다음 달 (덜 부정적이면 덜 떨어지나)
for col, lab in [("esg_neg_tone", "부정 ESG 기사 논조 IC"), ("esg_pos_tone", "긍정 ESG 기사 논조 IC")]:
    x = u[u[col].notna()]
    for nm_, a, b in PER:
        s = icser(x, col, "y", a, b); rows.append({"비교": lab, "기간": nm_, "다음달 차이(%p)": s.mean(), "t": tt(s)})
E = pd.DataFrame(rows); E.to_csv(f"{P}/esg_by_period.csv", index=False, encoding="utf-8-sig"); print(E.round(3).to_string(index=False))
# (2) H4: 논조 상위 20%(호재) 종목 중 개인 순매수 상위 절반 − 하위 절반, 나머지 종목에서의 같은 차이와 비교
x = u[u.s_all.notna() & u.retail_nb.notna()].copy()
x["good"] = x.groupby("month").s_all.rank(pct=True) > 0.8
x["rb"] = x.groupby(["month", "good"]).retail_nb.rank(pct=True) > 0.5
g = x.groupby(["month", "good", "rb"]).target.mean().unstack(["good", "rb"]) * 100
dg = g[(True, True)] - g[(True, False)]; do = g[(False, True)] - g[(False, False)]; dd = (dg - do).dropna()
rows = []
for nm_, a, b in PER:
    f_ = lambda s: s[(s.index >= a) & (s.index <= b)].dropna()
    rows.append({"기간": nm_, "호재 종목: 개인 많이−적게": f_(dg).mean(), "t1": tt(f_(dg)), "나머지: 개인 많이−적게": f_(do).mean(), "t2": tt(f_(do)), "차이의 차이": f_(dd).mean(), "t": tt(f_(dd))})
H = pd.DataFrame(rows); H.to_csv(f"{P}/h4_sent_retail.csv", index=False, encoding="utf-8-sig"); print(H.round(3).to_string(index=False))
