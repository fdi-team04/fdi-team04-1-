"""실적·애널리스트 테마 기사의 '논조'(감성 점수) 회사×월 집계 — 이미 계산한 감성 점수(sent_ckpt) 재사용, GPU 불필요 (PC 1~3분)
왜: 테마 판정은 '기사 있음/없음'이었음. 그 기사들이 얼마나 좋게/나쁘게 쓰였는지(KR-FinBert 점수)로도 다음 달 효과가 나는지 확인
준비 (같은 폴더):
  - 이 파일, theme_rules.py
  - articles_clean.csv  (코랩에서 감성 점수 돌릴 때 쓴 것과 같은 파일)
  - sent_ckpt 폴더 통째로 (구글 드라이브 감성 작업 폴더에 있음: c000000.npy ~ c000064.npy 같은 파일들)
실행: python 영섭_테마논조.py articles_clean.csv
결과(원문 없음 → Claude에게 보내기): theme_tone_monthly.csv
"""
import os, sys, glob, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theme_rules import themes_of
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]; TEST = "--test" in sys.argv
SRC = ARGS[0] if ARGS else "articles_clean.csv"; OUT = os.path.dirname(os.path.abspath(SRC))
CK = os.path.join(OUT, "sent_ckpt_test" if TEST else "sent_ckpt"); t0 = time.time()
files = sorted(glob.glob(os.path.join(CK, "c*.npy")))
if not files: sys.exit(f"감성 점수 폴더가 없어요: {CK}  → 드라이브의 sent_ckpt 폴더를 이 폴더 안에 통째로 넣어 주세요")
# 1. 감성 점수 돌릴 때와 '똑같은 순서'로 읽기 (그래야 점수가 맞는 기사에 붙음)
head = pd.read_csv(SRC, nrows=0).columns
pick = lambda *c: next((x for x in c if x in head), None)
C = {"id": pick("id", "nid", "news_id", "뉴스 식별자"), "code": pick("code", "종목코드"), "date": pick("date", "일자"),
     "title": pick("title", "제목"), "body": pick("body", "본문")}
x = pd.read_csv(SRC, usecols=[v for v in C.values() if v], dtype=str, nrows=2000 if TEST else None)
x = x.rename(columns={v: k for k, v in C.items() if v})
if "id" not in x: x["id"] = [f"r{i}" for i in range(len(x))]
x["code"] = x.code.str.replace(r"\.0$", "", regex=True).str.zfill(6)
x["date"] = pd.to_datetime(x.date.str[:10].str.replace(r"^(\d{4})(\d{2})(\d{2})$", r"\1-\2-\3", regex=True), errors="coerce")
x = x[x.date.notna()].reset_index(drop=True)
u = x.drop_duplicates("id")[["id"]].reset_index(drop=True)
s = np.concatenate([np.load(f) for f in files])
if len(s) != len(u):
    sys.exit(f"점수 개수({len(s):,})와 기사 수({len(u):,})가 달라요 → 코랩에서 쓴 articles_clean.csv와 다른 파일이거나 sent_ckpt가 덜 끝났어요")
u["s"] = s; x = x.merge(u, on="id", how="left")
print(f"기사 {len(x):,}건에 감성 점수 연결 ({time.time()-t0:.0f}초)")
# 2. 테마 표시 (제목 키워드, theme_rules.py와 같은 규칙)
th = x.title.map(themes_of)
G = {"실적": (["실적_호재"], ["실적_악재"]), "애널": (["목표가_상향"], ["목표가_하향"])}
for g, (pos, neg) in G.items():
    x[f"{g}_pos"] = th.map(lambda l: any(k in l for k in pos)); x[f"{g}_neg"] = th.map(lambda l: any(k in l for k in neg))
    x[g] = x[f"{g}_pos"] | x[f"{g}_neg"]
# 3. 월 배정 (그달 마지막 거래일 당일·이후 기사는 다음 달) → 회사×월
last = (x.date + pd.offsets.BMonthEnd(0)).dt.normalize(); per = x.date.dt.to_period("M")
x["month"] = per.where(x.date < last, per + 1).astype(str); x = x[x.month <= "2021-12"]
keys = ["month", "code"]; mon = x.groupby(keys).size().rename("n_all").to_frame()
for g in G:
    for col in [g, f"{g}_pos", f"{g}_neg"]:
        sub = x[x[col]].groupby(keys).s; mon[f"n_{col}"] = sub.size(); mon[f"tone_{col}"] = sub.mean()
for c in [c for c in mon if c.startswith("n_")]: mon[c] = mon[c].fillna(0).astype(int)
mon = mon.reset_index(); name = "theme_tone_monthly_test.csv" if TEST else "theme_tone_monthly.csv"
mon.to_csv(os.path.join(OUT, name), index=False, encoding="utf-8-sig")
print({c: int(mon[c].sum()) for c in mon if c.startswith("n_")})
print(f"끝! {name} {mon.shape} ({time.time()-t0:.0f}초) → Claude에게 보내기 (원문 없음)")
