"""테마 여러 개를 한 번에: 회사×월로 테마별 기사 수 세기 (VS Code, 1~3분)
같은 폴더: 이 파일, theme_rules.py, articles_clean.csv
실행: python 영섭_테마스크리닝.py articles_clean.csv
결과(원문 없음 → Claude에게 보내기): theme_monthly.csv  (month, code, n_all, 테마별 기사 수)
"""
import os, sys, time, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theme_rules import THEMES, themes_of
SRC = next((a for a in sys.argv[1:] if not a.startswith("--")), "articles_clean.csv")
OUT = os.path.dirname(os.path.abspath(SRC)); t0 = time.time()
head = pd.read_csv(SRC, nrows=0).columns
pick = lambda *c: next((x for x in c if x in head), None)
C = {"code": pick("code", "종목코드"), "date": pick("date", "일자"), "title": pick("title", "제목")}
x = pd.read_csv(SRC, usecols=list(C.values()), dtype=str).rename(columns={v: k for k, v in C.items()})
x["code"] = x.code.str.replace(r"\.0$", "", regex=True).str.zfill(6)
x["date"] = pd.to_datetime(x.date.str[:10].str.replace(r"^(\d{4})(\d{2})(\d{2})$", r"\1-\2-\3", regex=True), errors="coerce")
x = x[x.date.notna()]
th = x.title.map(themes_of)
for k in THEMES: x[k] = th.map(lambda l: k in l)
last = (x.date + pd.offsets.BMonthEnd(0)).dt.normalize(); per = x.date.dt.to_period("M")
x["month"] = per.where(x.date < last, per + 1).astype(str)       # 마지막 거래일 당일·이후 기사는 다음 달
x = x[x.month <= "2021-12"]
mon = x.groupby(["month", "code"]).agg(n_all=("title", "size"), **{k: (k, "sum") for k in THEMES}).reset_index()
mon.to_csv(os.path.join(OUT, "theme_monthly.csv"), index=False, encoding="utf-8-sig")
print("테마별 기사 수:", {k: int(mon[k].sum()) for k in THEMES})
print(f"끝! theme_monthly.csv {mon.shape} ({time.time()-t0:.0f}초) → Claude에게 보내기 (원문 없음)")
