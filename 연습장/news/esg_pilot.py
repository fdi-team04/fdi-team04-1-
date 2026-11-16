"""11/16 교수님 피드백 → ESG 틀로 좁히기: 파일럿 데이터로 먼저 찍어 보기
ESG 부정(논란): 수사·소송·사고·파업·산재·갑질·환경오염… / ESG 긍정(홍보): 사회공헌·기부·친환경·상생…
그 외(공시·실적·주가·신제품)는 '무관'으로 뺌. 회사×월 지표:
  esg_net = (긍정 건수 − 부정 건수) / (긍정 + 부정)   (−1 ~ +1, ESG 기사가 없으면 비움)
  contro  = 부정 기사 있음(1/0),  pr = 긍정 기사 있음(1/0)
같은 달·다음 달 수익률과의 관계 + '부정 있는 달 vs 없는 달' 다음 달 수익률 차이(종목 고정, 같은 달 비교)
"""
import re, numpy as np, pandas as pd, json
O = "/home/user/fdi-team04-1-/연습장/out"; P = f"{O}/news_pilot"
def tt(x): x = pd.Series(x).dropna(); return x.mean() / (x.std() / np.sqrt(len(x)))
NEG = r"횡령|배임|산재|중대재해|과로사|사망사고|사망|불매|소송|피소|고소|고발|오염|유출|압수수색|검찰|경찰|수사|구속|영장|기소|의혹|과징금|제재|적발|리콜|결함|파업|노조|갑질|폭언|담합|사고|화재|폭발|논란|징계|벌금|공정위|부당|불공정|비자금|분식|탈세|땅콩|성희롱|성추행|하청|불법"
POS = r"사회공헌|기부|후원|봉사|장학|상생|기금 전달|성금|나눔|복지|친환경|탄소중립|탄소 감축|재생에너지|ESG|윤리경영|투명경영|지배구조 개선|사외이사|동반성장|일자리 창출|채용 확대|캠페인|플로깅|기후"
OTHER = r"목표가|목표주가|투자의견|특징주|주가|급등|급락|상한가|하한가|신고가|[-–]\s*\S*(證|증권)\s*$"
a = pd.read_parquet("/home/user/data/news_pilot_work/articles_scored.parquet")
t = a.title.fillna("")
a["neg"] = t.str.contains(NEG) & ~t.str.contains(OTHER)
a["pos"] = t.str.contains(POS) & ~t.str.contains(OTHER) & ~a.neg
last = (a.date + pd.offsets.BMonthEnd(0)).dt.normalize(); per = a.date.dt.to_period("M")
a["month"] = per.where(a.date < last, per + 1).astype(str)
g = a.groupby(["month", "code"])
m = pd.DataFrame({"n_all": g.size(), "n_neg": g.neg.sum(), "n_pos": g.pos.sum()}).reset_index()
m["esg_n"] = m.n_neg + m.n_pos
m["esg_net"] = np.where(m.esg_n > 0, (m.n_pos - m.n_neg) / m.esg_n.replace(0, np.nan), np.nan)
print("ESG 관련 기사 비율:", round((a.neg | a.pos).mean(), 3), "| 부정", round(a.neg.mean(), 3), "긍정", round(a.pos.mean(), 3))
f = pd.read_parquet(f"{O}/features.parquet"); f["year"] = f.month.str[:4]
cy = m.assign(year=m.month.str[:4])[["code", "year"]].drop_duplicates()
u = f.merge(cy, on=["code", "year"]).merge(m, on=["month", "code"], how="left")
for c in ["n_all", "n_neg", "n_pos", "esg_n"]: u[c] = u[c].fillna(0)
u = u[u.target.notna()].copy()
u["contro"] = (u.n_neg > 0).astype(float); u["pr"] = (u.n_pos > 0).astype(float)
u["y"] = u.groupby("month").target.rank(pct=True); u["ret_r"] = u.groupby("month").ret.rank(pct=True)
rows = []
for col, lab in [("esg_net", "ESG 순지표 (ESG 기사 있는 칸만)"), ("contro", "부정 ESG 기사 있음"), ("pr", "긍정 ESG(홍보) 기사 있음")]:
    x = u[u[col].notna()]
    c0 = x.groupby("month").apply(lambda z: z[col].rank().corr(z.ret_r.rank()) if z[col].nunique() > 1 and len(z) >= 8 else np.nan, include_groups=False).dropna()
    c1 = x.groupby("month").apply(lambda z: z[col].rank().corr(z.y.rank()) if z[col].nunique() > 1 and len(z) >= 8 else np.nan, include_groups=False).dropna()
    rows.append({"변수": lab, "같은달 IC": c0.mean(), "t(같은달)": tt(c0), "다음달 IC": c1.mean(), "t(다음달)": tt(c1), "달수": len(c1), "칸": int(x[col].notna().sum())})
R = pd.DataFrame(rows); print(R.round(3).to_string(index=False))
# 부정 있는 종목 vs 없는 종목: 같은 달 안에서 다음 달 수익률 차이 (%p)
d = u.groupby(["month", "contro"]).target.mean().unstack() * 100; dd = (d[1] - d[0]).dropna()
d2 = u.groupby(["month", "pr"]).target.mean().unstack() * 100; dd2 = (d2[1] - d2[0]).dropna()
print(f"부정 ESG 있음−없음 다음 달 수익률: {dd.mean():.3f}%p (t {tt(dd):.2f}, {len(dd)}달)")
print(f"긍정 ESG 있음−없음 다음 달 수익률: {dd2.mean():.3f}%p (t {tt(dd2):.2f}, {len(dd2)}달)")
# 덮기: 부정이 있는 달에 홍보도 있으면?
both = u[u.contro == 1]; b = both.groupby(["month", "pr"]).target.mean().unstack() * 100; bb = (b[1] - b[0]).dropna()
print(f"부정 있는 종목 중 홍보도 있음−없음: {bb.mean():.3f}%p (t {tt(bb):.2f}, {len(bb)}달)")
R.to_csv(f"{P}/esg_pilot_ic.csv", index=False, encoding="utf-8-sig")
json.dump({"부정있음-없음": [dd.mean(), tt(dd)], "홍보있음-없음": [dd2.mean(), tt(dd2)], "덮기": [bb.mean(), tt(bb)]}, open(f"{P}/esg_pilot_diff.json", "w"), ensure_ascii=False)
print(a[a.neg].title.sample(8, random_state=1).tolist()); print(a[a.pos].title.sample(8, random_state=1).tolist())
