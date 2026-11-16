"""이안이 옛 이름으로 다시 받은 빅카인즈 파일(54개)을 정리해서 아영 news_monthly_200(기사 수)에 합침

왜: 이름이 바뀐 16개 회사는 옛 이름 시절 기사가 0건으로 들어가 있었음 → '기사 없음'으로 잘못 계산됨.
방법: 파일럿과 같은 정리(분석제외·언론사 5곳·중복 제거) → match_rules v2로 제목 매칭 →
      그 줄의 '목표 종목'에 붙은 기사만 셈 (다른 회사 기사는 아영 데이터에 이미 있으므로 이중 집계 방지)
      → 회사×월 기사 수를, 아영 데이터에서 그 종목-월이 비어 있을 때만 채움.
"""
import os, sys, glob, re, warnings, json, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from match_rules import build_index, match_title
J = pd.read_excel(os.path.join(os.path.dirname(__file__), "뉴스재수집_옛이름_17건.xlsx"), dtype=str)
row2code = {int(r.순번) + 1: r.종목코드 for r in J.itertuples()}
COLS = {"뉴스 식별자": "nid", "일자": "date", "언론사": "outlet", "제목": "title", "분석제외 여부": "excl"}
parts = []
for f in sorted(glob.glob("/home/user/data/oldname/*.xlsx")):
    row = int(os.path.basename(f)[3:5])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try: d = pd.read_excel(f, usecols=lambda c: c in COLS, dtype=str, engine="calamine")
        except Exception as e: print("못 읽음", f, e); continue
    parts.append(d.rename(columns=COLS).assign(row=row, target=row2code[row]))
raw = pd.concat(parts, ignore_index=True); log = {"읽은 기사": len(raw)}
d = raw[~raw.excl.fillna("").str.contains("중복|예외")]
d = d[d.outlet.isin(["매일경제", "머니투데이", "서울경제", "파이낸셜뉴스", "한국경제"])]
d["date"] = pd.to_datetime(d.date.str[:8], format="%Y%m%d", errors="coerce"); d = d.dropna(subset=["date"])
d = d.drop_duplicates(["nid", "target"]).drop_duplicates(["title", "outlet", "date", "target"]); log["정리 후"] = len(d)
c = pd.read_csv(os.path.join(os.path.dirname(__file__), "company_list_최종.csv"), dtype=str, keep_default_na=False)
idx = build_index(c); ns = sorted(idx, key=len, reverse=True)
d["ym"] = d.date.dt.strftime("%Y-%m")
d["hit"] = [t in match_title(ti, y, idx, ns) for ti, y, t in zip(d.title, d.ym, d.target)]
k = d[d.hit].copy(); log["목표 종목 제목 매칭"] = len(k)
last = (k.date + pd.offsets.BMonthEnd(0)).dt.normalize(); per = k.date.dt.to_period("M")
k["month"] = per.where(k.date < last, per + 1).astype(str)
add = k.groupby(["target", "month"]).size().rename("n_add").reset_index().rename(columns={"target": "code"})
by = add.assign(y=add.month.str[:4]).groupby(["code", "y"]).n_add.sum().unstack(fill_value=0)
print(by.to_string())
nm = pd.read_csv("/root/.claude/uploads/03b15712-f400-5188-9b23-fea64cb12856/8515ff1c-news_monthly_200.csv", dtype={"code": str}); nm["code"] = nm.code.str.zfill(6)
mm = nm.merge(add, on=["code", "month"], how="outer")
before = mm.n_articles.notna() & mm.n_add.notna()
log["아영 데이터에 이미 있던 칸(그대로 둠)"] = int(before.sum())
mm["n_articles"] = mm.n_articles.fillna(mm.n_add); mm["source"] = (mm.n_add.notna() & ~before).map({True: "옛이름추가", False: "아영"})
mm["name"] = mm.groupby("code").name.transform(lambda s: s.ffill().bfill())
mm = mm.drop(columns="n_add").sort_values(["code", "month"])
O = "/home/user/fdi-team04-1-/연습장/out/news_full200"; os.makedirs(O, exist_ok=True)
mm.to_csv(f"{O}/news_monthly_200_plus.csv", index=False, encoding="utf-8-sig")
log["추가된 종목-월"] = int((mm.source == "옛이름추가").sum()); log["추가된 기사 수"] = int(mm.loc[mm.source == "옛이름추가", "n_articles"].sum())
json.dump(log, open(f"{O}/oldname_merge_log.json", "w"), ensure_ascii=False, indent=1); print(log)
