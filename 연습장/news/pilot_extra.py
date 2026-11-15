"""뉴스 파일럿 3: '같은 달엔 맞는데 다음 달엔 안 맞는다'를 더 쪼개 보기

pilot_analyze 결과: 논조와 같은 달 수익률은 뚜렷한 + (IC 0.095, t 8.4), 다음 달은 0 (IC 0.001).
제목을 눈으로 보니 [특징주]·'주가 4%↓'처럼 이미 움직인 주가를 전하는 기사가 많다 → 같은 달 상관의 일부는
'뉴스가 주가를 움직인 것'이 아니라 '주가가 뉴스를 만든 것'일 수 있다. 그래서 아래를 확인한다.
 A. 주가 보도 기사(특징주·주가·급등락·%↑↓·목표가)를 빼고 다시 → 같은 달 상관이 남나, 다음 달은?
 B. 월말 10일 기사만 → 반영이 덜 된 늦은 소식이 다음 달에 이어지나
 C. 논조 '변화' = 이번 달 논조 − 그 회사 직전 12개월 평균 → 평소보다 좋은 소식인지
 D. 비율형: 강한 부정(s<−0.5) 비율, 강한 긍정(s>0.5) 비율 → 점수가 ±1에 몰려 평균이 정보를 잃는지
 E. 화제형 개수 (n_topic, 기사 수 효과를 뺀 비율)
"""
import re, numpy as np, pandas as pd
O = "/home/user/fdi-team04-1-/연습장/out"; P = f"{O}/news_pilot"
def tt(x): x = pd.Series(x).dropna(); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 else np.nan
a = pd.read_parquet("/home/user/data/news_pilot_work/articles_scored.parquet")
a = a[a.s.notna()].copy()
last = (a.date + pd.offsets.BMonthEnd(0)).dt.normalize(); per = a.date.dt.to_period("M")
a["month"] = per.where(a.date < last, per + 1).astype(str); a = a[a.month <= "2021-12"]
PRICE = re.compile(r"특징주|관심주|주가|급등|급락|강세|약세|상한가|하한가|신고가|신저가|목표가|[↑↓]|%\s*(?:상승|하락|올|내)|오름세|내림세|반등")
a["price_news"] = a.title.str.contains(PRICE)
a["late"] = (pd.PeriodIndex(a.month, freq="M").to_timestamp(how="end").normalize().values - a.date.values) <= np.timedelta64(14, "D")   # 배정된 달의 마지막 2주(≈10영업일)
print("주가 보도 기사 비율:", round(a.price_news.mean(), 3))

def agg(x, name):
    g = x.groupby(["month", "code"]).s
    return pd.DataFrame({name: g.mean()})
V = [agg(a, "all"), agg(a[~a.price_news], "noprice"), agg(a[a.price_news], "price_only"), agg(a[a.late], "late"),
     agg(a[a.late & ~a.price_news], "late_noprice")]
g = a.groupby(["month", "code"])
V.append(pd.DataFrame({"neg_share": g.s.apply(lambda s: (s < -0.5).mean()), "pos_share": g.s.apply(lambda s: (s > 0.5).mean()),
                       "n": g.size(), "topic_share": g.type.apply(lambda t: (t == "화제형").mean())}))
M = pd.concat(V, axis=1).reset_index().sort_values(["code", "month"])
# C. 변화: 직전 12개 '기사 있던 달' 평균 대비 (미래 정보 없음: shift 1)
M["surprise"] = M["all"] - M.groupby("code")["all"].transform(lambda s: s.shift(1).rolling(12, min_periods=3).mean())

f = pd.read_parquet(f"{O}/features.parquet")
u = f.merge(M, on=["month", "code"]); u = u[u.target.notna()]
u["ret_r"] = u.groupby("month").ret.rank(pct=True); u["y"] = u.groupby("month").target.rank(pct=True)
def mic(x, col, ycol):
    return x[x[col].notna()].groupby("month").apply(lambda z: z[col].rank().corr(z[ycol].rank()) if len(z) >= 8 else np.nan, include_groups=False).dropna()
rows = []
for col, nm in [("all", "전체 논조(기준)"), ("noprice", "A. 주가 보도 뺀 논조"), ("price_only", "A'. 주가 보도 기사만"),
                ("late", "B. 월말 기사 논조"), ("late_noprice", "B'. 월말·주가보도 제외"), ("surprise", "C. 논조 변화(평소 대비)"),
                ("neg_share", "D. 강한 부정 비율"), ("pos_share", "D. 강한 긍정 비율"), ("topic_share", "E. 화제형 비율")]:
    for lab, s, e in [("전체", "2001", "2022"), ("2011–21", "2011", "2022")]:
        x = u[(u.month >= s) & (u.month < e)]
        c0, c1 = mic(x, col, "ret_r"), mic(x, col, "y")
        rows.append({"변수": nm, "기간": lab, "같은달 IC": c0.mean(), "t(같은달)": tt(c0), "다음달 IC": c1.mean(), "t(다음달)": tt(c1), "칸 수": int(x[col].notna().sum())})
E = pd.DataFrame(rows); E.to_csv(f"{P}/pilot_extra_ic.csv", index=False, encoding="utf-8-sig")
print(E.round(3).to_string(index=False))
