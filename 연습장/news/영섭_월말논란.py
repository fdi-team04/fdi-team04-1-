"""H5-B '그달 내내 이어진 논란' 확인용: 회사×월로 부정 ESG 기사를 '그달 앞쪽 / 마지막 7일'로 나눠 셈 (VS Code, 1~3분)

왜: 월 합계만으로는 기사가 월초에 몰렸는지 월말까지 이어졌는지 모른다.
    '월말(마지막 7일)에도 부정 기사가 있다' = 이달 말 시점에 논란이 아직 진행 중 → 다음 달에도 영향이 남는지 보려는 것.
    월말 7일도 이달 말까지 알 수 있는 정보라 예측 규칙에 어긋나지 않음.
준비: 같은 폴더에 이 파일, esg_rules.py(v1.2), articles_clean.csv
실행: python 영섭_월말논란.py articles_clean.csv
결과(원문 없음 → Claude에게 보내기): esg_late_monthly.csv  (month, code, n_all, n_neg, n_neg_late, n_neg_early, neg_days)
"""
import os, sys, time
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esg_rules import classify_esg
SRC = next((a for a in sys.argv[1:] if not a.startswith("--")), "articles_clean.csv")
OUT = os.path.dirname(os.path.abspath(SRC)); t0 = time.time()
head = pd.read_csv(SRC, nrows=0).columns
pick = lambda *c: next((x for x in c if x in head), None)
C = {"code": pick("code", "종목코드"), "date": pick("date", "일자"), "title": pick("title", "제목")}
x = pd.read_csv(SRC, usecols=list(C.values()), dtype=str).rename(columns={v: k for k, v in C.items()})
x["code"] = x.code.str.replace(r"\.0$", "", regex=True).str.zfill(6)
x["date"] = pd.to_datetime(x.date.str[:10].str.replace(r"^(\d{4})(\d{2})(\d{2})$", r"\1-\2-\3", regex=True), errors="coerce")
x = x[x.date.notna()]
x["neg"] = x.title.fillna("").map(lambda t: classify_esg(t)[0] == "부정")
# 월 배정: 그달 마지막 거래일 당일·이후 기사는 다음 달 (기존 분석과 같은 규칙)
last = (x.date + pd.offsets.BMonthEnd(0)).dt.normalize(); per = x.date.dt.to_period("M")
x["month"] = per.where(x.date < last, per + 1)
cut = (x.month.dt.to_timestamp(how="end") + pd.offsets.BMonthEnd(0)).dt.normalize()   # 그 배정 월의 마지막 거래일(이 날부터는 다음 달)
x["late"] = (cut - x.date.dt.normalize()).dt.days.between(1, 7)                      # 마감 전 7일 안
x["month"] = x.month.astype(str); x = x[x.month <= "2021-12"]
g = x.groupby(["month", "code"])
mon = pd.DataFrame({"n_all": g.size(), "n_neg": g.neg.sum(),
                    "n_neg_late": x[x.neg & x.late].groupby(["month", "code"]).size(),
                    "neg_days": x[x.neg].groupby(["month", "code"]).date.nunique()}).fillna(0).astype(int)
mon["n_neg_early"] = mon.n_neg - mon.n_neg_late
mon.reset_index().to_csv(os.path.join(OUT, "esg_late_monthly.csv"), index=False, encoding="utf-8-sig")
print(f"끝! esg_late_monthly.csv {mon.shape} ({time.time()-t0:.0f}초) → Claude에게 보내기 (원문 없음)")
