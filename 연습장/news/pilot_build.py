"""뉴스 파일럿 1: 원자료 52개 파일 → 정리 → 회사 매칭(기간 반영) → 유형 분류 → 감성 점수 → 회사×월

왜 파일럿인가: 정리본(128만 건, 686MB)은 드라이브 도구 한도(10MB)를 넘어 이 환경으로 못 가져온다.
그래서 같은 빅카인즈 원자료 중 10MB 미만 파일 52개(2001~2021, 담당 1~4)를 받아 같은 절차를 처음부터 적용했다.
목적: ① 이름 겹침 문제를 '기간'으로 푸는 방법을 실제로 시험 ② KR-FinBert-SC 점수가 수익률과 관계가 있는지 미리 보기.
원문은 data 폴더(깃 밖)에만 두고, 깃에는 원문이 없는 집계표만 남긴다.
"""
import os, re, glob, warnings, time, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from news_rules import classify

RAW = "/home/user/data/news_pilot"
WORK = "/home/user/data/news_pilot_work"; os.makedirs(WORK, exist_ok=True)
OUT = "/home/user/fdi-team04-1-/연습장/out/news_pilot"; os.makedirs(OUT, exist_ok=True)
LOG = {}

# ---------- 1. 읽기 ----------
COLS = {"뉴스 식별자": "nid", "일자": "date", "언론사": "outlet", "제목": "title", "본문": "body", "분석제외 여부": "excl"}
parts = []
for f in sorted(glob.glob(f"{RAW}/*.xlsx")):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = pd.read_excel(f, usecols=lambda c: c in COLS, dtype=str, engine="calamine")
    d = d.rename(columns=COLS); d["file"] = os.path.basename(f); parts.append(d)
raw = pd.concat(parts, ignore_index=True)
LOG["1_읽은 기사"] = len(raw); LOG["파일 수"] = len(parts)

# ---------- 2. 정리 ----------
d = raw[~raw.excl.fillna("").str.contains("중복|예외")]; LOG["2a_분석제외 뺀 뒤"] = len(d)
d = d[d.outlet.isin(["매일경제", "머니투데이", "서울경제", "파이낸셜뉴스", "한국경제"])]; LOG["2b_언론사 5곳"] = len(d)
d["date"] = pd.to_datetime(d.date.astype(str).str[:8], format="%Y%m%d", errors="coerce")
d = d.dropna(subset=["date"]); d = d[(d.date >= "2000-12-01") & (d.date <= "2021-12-31")]
d["title"] = d.title.fillna("").str.replace("車", "차"); d["body"] = d.body.fillna("")
d = d.drop_duplicates("nid"); LOG["2c_뉴스식별자 중복 뺀 뒤"] = len(d)
d = d.drop_duplicates(["title", "outlet", "date"]); LOG["2d_제목·언론사·날짜 중복 뺀 뒤"] = len(d)

# ---------- 3. 회사 매칭 (제목, 기간 반영) ----------
comp = pd.read_excel("/home/user/fdi-team04-1-/연습장/news/뉴스수집_검색목록_상위200.xlsx", sheet_name="회사·옛이름 (478)", dtype=str)
names = []
for _, r in comp.iterrows():
    for t in str(r["검색어 (복사용)"]).split(" OR "):
        t = t.strip().replace("車", "차")
        if len(t) >= 2: names.append((t, r["종목코드"]))
names = pd.DataFrame(names, columns=["name", "code"]).drop_duplicates()
# 종목이 실제로 거래되던 달(수익률 있는 달) = 이름을 쓸 수 있는 기간의 기본값
f = pd.read_parquet("/home/user/fdi-team04-1-/연습장/out/features.parquet", columns=["month", "code"])
active = f.groupby("code").month.agg(["min", "max"])
# 이름 겹침 수동 규칙 (합병·분할로 이름이 옮겨간 경우). 확인 필요 표시 = 공시 날짜로 재확인할 것
OVERRIDE = {
    "SK": [("003600", None, "2015-07"), ("034730", "2015-08", None)],
    "삼성물산": [("000830", None, "2015-08"), ("028260", "2015-09", None)],
    "미래에셋증권": [("037620", None, "2016-12"), ("006800", "2021-03", None)],
    "대우증권": [("006800", None, "2016-11")], "미래에셋대우": [("006800", "2016-11", "2021-02")],
    "한국타이어": [("000240", None, "2012-08"), ("161390", "2012-09", "2019-05")],
    "오리온": [("001800", None, "2017-05"), ("271560", "2017-06", None)],
    "녹십자": [("005250", None, "2004-06"), ("006280", "2004-07", None)],
    "하이트맥주": [("000140", None, "2008-06"), ("103150", "2008-07", "2011-08")],
    "롯데제과": [("004990", None, "2017-09"), ("280360", "2017-10", "2022-03")],
    "NHN": [("035420", None, "2013-07"), ("181710", "2019-03", None)],
    "태영건설": [("009410", None, "2020-08"), ("363280", "2020-09", None)],
    "우리금융지주": [("053000", None, "2014-10"), ("316140", "2019-02", None)],
    "BGF리테일": [("027410", None, "2017-10"), ("282330", "2017-11", None)],
    "두산건설": [("011160", None, "2008-12"), ("002950", "2009-01", None)],
    "하나은행": [("002860", None, "2002-11"), ("007360", None, "2002-11")],
}
ov_names = set(OVERRIDE)
by_name = names.groupby("name").code.apply(list).to_dict()
alts = sorted(by_name, key=len, reverse=True)
pat = re.compile("|".join((r"(?<![A-Za-z0-9&])" + re.escape(n) + r"(?![A-Za-z0-9&])") if re.search(r"[A-Za-z0-9]", n) else re.escape(n) for n in alts))
d["ym"] = d.date.dt.strftime("%Y-%m")
stat = {"모호해서 버림": 0, "기간 규칙으로 배정": 0}
def owners(name, ym):
    if name in ov_names:
        hit = [c for c, a, b in OVERRIDE[name] if (a is None or ym >= a) and (b is None or ym <= b)]
        if hit: stat["기간 규칙으로 배정"] += 1
        return hit[:1] if len(hit) == 1 else []
    cs = by_name[name]
    if len(cs) == 1: return cs
    act = [c for c in cs if c in active.index and active.loc[c, "min"] <= ym <= active.loc[c, "max"]]
    if len(act) == 1:
        stat["기간 규칙으로 배정"] += 1; return act
    stat["모호해서 버림"] += 1; return []
rows = []
for nid, t, ym in zip(d.nid, d.title, d.ym):
    got = set()
    for m in pat.finditer(t):
        got.update(owners(m.group(0), ym))
    for c in got: rows.append((nid, c))
mp = pd.DataFrame(rows, columns=["nid", "code"])
a = d.merge(mp, on="nid")
LOG["3_제목 매칭된 (기사,회사)"] = len(a); LOG["3_제목 매칭된 기사"] = a.nid.nunique(); LOG["3_매칭 종목 수"] = a.code.nunique()
LOG.update({f"3_{k}": v for k, v in stat.items()})

# ---------- 4. 분류 ----------
cl = a.drop_duplicates("nid")[["nid", "title", "body"]].copy()
cl[["type", "복합", "근거"]] = cl.apply(lambda r: classify(r.title, r.body), axis=1, result_type="expand")
a = a.merge(cl[["nid", "type", "복합"]], on="nid")
LOG["4_유형 비율"] = a.drop_duplicates("nid").type.value_counts(normalize=True).round(3).to_dict()

# ---------- 5. 감성 점수 (KR-FinBert-SC, 재학습 없음) ----------
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("snunlp/KR-FinBert-SC")
mdl = AutoModelForSequenceClassification.from_pretrained("snunlp/KR-FinBert-SC").eval()
torch.set_num_threads(os.cpu_count())
lab = {v.lower(): k for k, v in mdl.config.id2label.items()}
u = a.drop_duplicates("nid")[["nid", "title", "body", "type"]]
u = u[u.type != "광고"]
texts = (u.title + " " + u.body.str[:200]).tolist(); S = []; t0 = time.time()
for i in range(0, len(texts), 64):
    enc = tok(texts[i:i + 64], padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        p = torch.softmax(mdl(**enc).logits, -1).numpy()
    S.append(np.c_[p[:, lab["positive"]], p[:, lab["negative"]]])
    if i % 6400 == 0: print(i, len(texts), f"{time.time()-t0:.0f}s", flush=True)
S = np.vstack(S)
u = u.assign(p_pos=S[:, 0], p_neg=S[:, 1], s=S[:, 0] - S[:, 1])
LOG["5_점수 매긴 기사(광고 제외)"] = len(u); LOG["5_걸린 시간(초, CPU)"] = round(time.time() - t0)
a = a.merge(u[["nid", "p_pos", "p_neg", "s"]], on="nid", how="left")
# 눈 점검용 (원문 포함 → data 폴더에만)
u.sort_values("s").head(30)[["title", "type", "s"]].to_csv(f"{WORK}/most_negative30.csv", index=False, encoding="utf-8-sig")
u.sort_values("s").tail(30)[["title", "type", "s"]].to_csv(f"{WORK}/most_positive30.csv", index=False, encoding="utf-8-sig")
a.drop(columns=["body"]).to_parquet(f"{WORK}/articles_scored.parquet")

# ---------- 6. 월 배정·회사×월 (원문 없음 → 깃에 남겨도 됨) ----------
last = (a.date + pd.offsets.BMonthEnd(0)).dt.normalize()
per = a.date.dt.to_period("M"); a["month"] = per.where(a.date < last, per + 1).astype(str)
a = a[a.month <= "2021-12"]
g = a.groupby(["month", "code"])
mon = g.size().rename("n_all").to_frame()
mon["all_tone"] = g.s.mean()
for t, k in [("화제형", "topic"), ("공시형", "disc"), ("논란형", "contro"), ("기타", "etc")]:
    sub = a[a.type == t].groupby(["month", "code"]); mon[f"n_{k}"] = sub.size(); mon[f"{k}_tone"] = sub.s.mean()
for c in [c for c in mon if c.startswith("n_")]: mon[c] = mon[c].fillna(0).astype(int)
mon = mon.reset_index(); mon.to_csv(f"{OUT}/pilot_news_monthly.csv", index=False, encoding="utf-8-sig")
# 어떤 회사·기간이 파일럿에 들어 있는가 (파일럿 범위)
cov = a.groupby("code").month.agg(["min", "max", "nunique"]).reset_index(); cov.to_csv(f"{OUT}/pilot_coverage.csv", index=False, encoding="utf-8-sig")
LOG["6_회사×월 칸"] = len(mon); LOG["6_종목"] = mon.code.nunique()
json.dump(LOG, open(f"{OUT}/pilot_build_log.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(LOG, ensure_ascii=False, indent=1))
