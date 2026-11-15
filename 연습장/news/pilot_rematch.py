"""뉴스 파일럿 4: 매칭 규칙 v2(match_rules.py)로 회사를 다시 붙이고 회사×월을 다시 만듦
왜: v1(pilot_build)은 'SK텔'→SK, '광고대상'→대상, '-대우' 리포트 꼬리표 → 대우처럼 잘못 붙는 경우가 (기사,회사)의 약 18%.
점수(s)와 유형은 기사 단위라 그대로 쓰고, 회사만 다시 붙인다. 결과는 pilot_build와 같은 파일 이름으로 덮어씀(v1은 _v1로 보관)
"""
import os, sys, shutil, json, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from match_rules import build_index, match_title
W = "/home/user/data/news_pilot_work"; O = "/home/user/fdi-team04-1-/연습장/out/news_pilot"
if not os.path.exists(f"{W}/articles_scored_v1.parquet"): shutil.copy(f"{W}/articles_scored.parquet", f"{W}/articles_scored_v1.parquet")
for f in ["pilot_news_monthly.csv", "pilot_coverage.csv"]:
    if not os.path.exists(f"{O}/{f[:-4]}_v1.csv"): shutil.copy(f"{O}/{f}", f"{O}/{f[:-4]}_v1.csv")
a = pd.read_parquet(f"{W}/articles_scored_v1.parquet")
c = pd.read_csv(os.path.join(os.path.dirname(__file__), "company_list_최종.csv"), dtype=str, keep_default_na=False)
idx = build_index(c); ns = sorted(idx, key=len, reverse=True)
u = a.drop(columns="code").drop_duplicates("nid").copy(); u["ym"] = u.date.dt.strftime("%Y-%m")
u["code"] = [match_title(t, y, idx, ns) for t, y in zip(u.title, u.ym)]
v = u.explode("code").dropna(subset=["code"]).drop(columns="ym")
v.to_parquet(f"{W}/articles_scored.parquet")
last = (v.date + pd.offsets.BMonthEnd(0)).dt.normalize(); per = v.date.dt.to_period("M")
v["month"] = per.where(v.date < last, per + 1).astype(str); v = v[v.month <= "2021-12"]
g = v.groupby(["month", "code"]); mon = g.size().rename("n_all").to_frame(); mon["all_tone"] = g.s.mean()
for t, k in [("화제형", "topic"), ("공시형", "disc"), ("논란형", "contro"), ("기타", "etc")]:
    sub = v[v.type == t].groupby(["month", "code"]); mon[f"n_{k}"] = sub.size(); mon[f"{k}_tone"] = sub.s.mean()
for col in [x for x in mon if x.startswith("n_")]: mon[col] = mon[col].fillna(0).astype(int)
mon.reset_index().to_csv(f"{O}/pilot_news_monthly.csv", index=False, encoding="utf-8-sig")
v.groupby("code").month.agg(["min", "max", "nunique"]).reset_index().to_csv(f"{O}/pilot_coverage.csv", index=False, encoding="utf-8-sig")
log = {"v1 (기사,회사)": len(a), "v2 (기사,회사)": len(v), "v2 기사": int(v.nid.nunique()), "v2 종목": int(v.code.nunique()), "v2 회사×월": len(mon)}
json.dump(log, open(f"{O}/pilot_rematch_log.json", "w"), ensure_ascii=False, indent=1); print(log)
