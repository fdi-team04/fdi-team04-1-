"""VS Code(내 컴퓨터)에서 돌리는 ESG 계산 — 코랩 대신

하는 일: 아영 정리본(articles_clean.csv) → ESG 부정/긍정/무관 분류(제목) → ESG 기사만 감성 점수 → 회사×월 집계
왜 ESG 기사만 점수: 129만 건 전부를 CPU로 점수 매기면 하루 가까이 걸림. ESG 분석에 필요한 건 ESG 기사(약 6%)뿐이라
                   이렇게 하면 노트북 CPU로도 1~2시간, 맥(M칩)이나 엔비디아 GPU면 훨씬 빠름.

준비 (터미널에서 한 번):
    pip install pandas transformers torch
같은 폴더에 둘 것: 이 파일, esg_rules.py, articles_clean.csv(아영 드라이브에서 내려받기)
실행:
    python 영섭_ESG_VSCode.py articles_clean.csv
    python 영섭_ESG_VSCode.py articles_clean.csv --no-score   ← 감성 점수 없이 ESG 건수만 (모델 400MB 안 받음, 1분)
결과(같은 폴더, 원문 없음 → Claude에게 보내기): esg_monthly.csv, esg_coverage.csv
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esg_rules import classify_esg

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
NO_SCORE = "--no-score" in sys.argv
SRC = ARGS[0] if ARGS else "articles_clean.csv"
OUT = os.path.dirname(os.path.abspath(SRC))

# 1. 읽기 (칸 이름이 조금 달라도 맞춰 읽음)
t0 = time.time()
head = pd.read_csv(SRC, nrows=0).columns
pick = lambda *c: next((x for x in c if x in head), None)
C = {"id": pick("id", "nid", "news_id", "뉴스 식별자"), "code": pick("code", "종목코드"), "date": pick("date", "일자"),
     "title": pick("title", "제목"), "body": pick("body", "본문")}
print("칸 연결:", C)
use = [v for v in C.values() if v]
x = pd.read_csv(SRC, usecols=use, dtype=str).rename(columns={v: k for k, v in C.items() if v})
if "id" not in x: x["id"] = [f"r{i}" for i in range(len(x))]
if "body" not in x: x["body"] = ""
x["code"] = x.code.str.replace(r"\.0$", "", regex=True).str.zfill(6)
x["date"] = pd.to_datetime(x.date.str[:10].str.replace(r"^(\d{4})(\d{2})(\d{2})$", r"\1-\2-\3", regex=True), errors="coerce")
x = x[x.date.notna()]
print(f"읽은 기사 {len(x):,}건, 종목 {x.code.nunique()}개 ({time.time()-t0:.0f}초)")

# 2. ESG 분류 (제목만)
t0 = time.time()
r = x.title.fillna("").map(classify_esg)
x["esg"] = r.str[0]; x["esg_area"] = r.str[1]
print("ESG 비율:", x.esg.value_counts(normalize=True).round(4).to_dict(), f"({time.time()-t0:.0f}초)")

# 3. 감성 점수: ESG 기사만 (KR-FinBert-SC, s = P(긍정) − P(부정))
if NO_SCORE:
    print("--no-score: 감성 점수는 건너뜀 (ESG 부정·긍정 건수와 순지표만 계산)")
    x["s"] = np.nan
else:
  import torch
  from transformers import AutoTokenizer, AutoModelForSequenceClassification
  dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
  tok = AutoTokenizer.from_pretrained("snunlp/KR-FinBert-SC")
  mdl = AutoModelForSequenceClassification.from_pretrained("snunlp/KR-FinBert-SC").eval().to(dev)
  lab = {v.lower(): k for k, v in mdl.config.id2label.items()}
  e = x[x.esg != "무관"].drop_duplicates("id")[["id", "title", "body"]].copy()
  texts = (e.title.fillna("") + " " + e.body.fillna("").str[:200]).tolist()
  print(f"점수 매길 ESG 기사 {len(texts):,}건 | 장치 {dev}")
  S, t0, B = [], time.time(), 64
  for i in range(0, len(texts), B):
      enc = tok(texts[i:i + B], padding=True, truncation=True, max_length=160, return_tensors="pt").to(dev)
      with torch.no_grad():
          p = torch.softmax(mdl(**enc).logits.float(), -1).cpu().numpy()
      S.append(p[:, lab["positive"]] - p[:, lab["negative"]])
      if (i // B) % 100 == 0:
          print(f"  {min(i + B, len(texts)):,}/{len(texts):,}  {time.time()-t0:.0f}초", flush=True)
  e["s"] = np.concatenate(S) if S else []
  x = x.merge(e[["id", "s"]], on="id", how="left")

# 4. 월 배정 (그달 마지막 거래일 당일·이후 기사는 다음 달) → 회사×월
last = (x.date + pd.offsets.BMonthEnd(0)).dt.normalize(); per = x.date.dt.to_period("M")
x["month"] = per.where(x.date < last, per + 1).astype(str)
x = x[x.month <= "2021-12"]
g = x.groupby(["month", "code"]); mon = g.size().rename("n_all").to_frame()
for lab_, k in [("부정", "esg_neg"), ("긍정", "esg_pos")]:
    sub = x[x.esg == lab_].groupby(["month", "code"]); mon[f"n_{k}"] = sub.size(); mon[f"{k}_tone"] = sub.s.mean()
for c in ["n_esg_neg", "n_esg_pos"]: mon[c] = mon[c].fillna(0).astype(int)
n = mon.n_esg_neg + mon.n_esg_pos
mon["esg_net"] = np.where(n > 0, (mon.n_esg_pos - mon.n_esg_neg) / n.where(n > 0), np.nan)
mon = mon.reset_index()
mon.to_csv(os.path.join(OUT, "esg_monthly.csv"), index=False, encoding="utf-8-sig")
cov = x.assign(year=x.date.dt.year).groupby(["code", "year"]).size().unstack(fill_value=0)
cov.to_csv(os.path.join(OUT, "esg_coverage.csv"), encoding="utf-8-sig")
print(f"끝! esg_monthly.csv {mon.shape}, esg_coverage.csv {cov.shape} → 이 두 파일을 Claude에게 보내기")
