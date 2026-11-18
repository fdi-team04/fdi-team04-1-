"""전체 기사(약 129만 건) 감성 점수 → 회사×월 집계  (교수님 F3 '뉴스도 먼저 찍어 봐라'를 전체 데이터로)

왜: 발표 9장 감성 결과(같은 달 IC 0.121, 다음 달 0.002)는 일부 데이터(52개 파일)로 낸 값이다.
    전체로 다시 재야 "파일럿이라 그런 것 아니냐"는 질문에 답할 수 있다. (기사 수 효과는 전체로 보니 사라졌었음)

준비 (한 번):  pip install pandas pyarrow transformers torch
같은 폴더: 이 파일, esg_rules.py, articles_clean.csv
실행:
    python 영섭_감성전체.py articles_clean.csv            ← 전체
    python 영섭_감성전체.py articles_clean.csv --test     ← 2,000건만 (속도 확인용, 1~2분)
중간에 꺼져도 OK: 같은 명령을 다시 치면 이어서 함 (sent_ckpt 폴더에 저장됨)
결과 (원문 없음 → Claude에게 보내기): sent_monthly.csv
   칸: month, code, n, s_all(전체 논조), n_noprice / s_noprice(주가 보도 기사 뺀 논조),
       pos_share / neg_share(강한 긍정·부정 비율), n_esg_neg / n_esg_pos / esg_neg_tone / esg_pos_tone
"""
import os, sys, time, re, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esg_rules import classify_esg

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
TEST = "--test" in sys.argv
SRC = ARGS[0] if ARGS else "articles_clean.csv"
OUT = os.path.dirname(os.path.abspath(SRC))
CK = os.path.join(OUT, "sent_ckpt_test" if TEST else "sent_ckpt"); os.makedirs(CK, exist_ok=True)
PRICE = re.compile(r"특징주|관심주|주가|급등|급락|강세|약세|상한가|하한가|신고가|신저가|목표가|[↑↓]|%\s*(?:상승|하락|올|내)|오름세|내림세|반등")

# 1. 읽기 (칸 이름이 조금 달라도 맞춰 읽음)
t0 = time.time()
head = pd.read_csv(SRC, nrows=0).columns
pick = lambda *c: next((x for x in c if x in head), None)
C = {"id": pick("id", "nid", "news_id", "뉴스 식별자"), "code": pick("code", "종목코드"), "date": pick("date", "일자"),
     "title": pick("title", "제목"), "body": pick("body", "본문")}
print("칸 연결:", C)
x = pd.read_csv(SRC, usecols=[v for v in C.values() if v], dtype=str, nrows=2000 if TEST else None)
x = x.rename(columns={v: k for k, v in C.items() if v})
if "id" not in x: x["id"] = [f"r{i}" for i in range(len(x))]
if "body" not in x: x["body"] = ""
x["code"] = x.code.str.replace(r"\.0$", "", regex=True).str.zfill(6)
x["date"] = pd.to_datetime(x.date.str[:10].str.replace(r"^(\d{4})(\d{2})(\d{2})$", r"\1-\2-\3", regex=True), errors="coerce")
x = x[x.date.notna()].reset_index(drop=True)
print(f"읽은 기사 {len(x):,}건, 종목 {x.code.nunique()}개 ({time.time()-t0:.0f}초)")

# 2. 감성 점수 (파일럿과 같은 방식: 제목 + 본문 앞 200자, s = P(긍정) − P(부정)). 같은 기사는 한 번만.
u = x.drop_duplicates("id")[["id", "title", "body"]].reset_index(drop=True)
texts = (u.title.fillna("") + " " + u.body.fillna("").str[:200]).tolist()
CHUNK = 20000                                          # 2만 건마다 저장 → 꺼져도 이어서
done = {int(os.path.basename(f)[1:7]) for f in glob.glob(os.path.join(CK, "c*.npy"))}
todo = [k for k in range(0, len(texts), CHUNK) if k // CHUNK not in done]
print(f"점수 매길 기사 {len(texts):,}건 | 저장된 묶음 {len(done)}개, 남은 묶음 {len(todo)}개")
if todo:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("snunlp/KR-FinBert-SC")
    mdl = AutoModelForSequenceClassification.from_pretrained("snunlp/KR-FinBert-SC").eval().to(dev)
    if dev == "cuda": mdl = mdl.half()                 # GPU면 반정밀도로 2배 빠르게
    if dev == "cpu": torch.set_num_threads(os.cpu_count() or 4)
    lab = {v.lower(): k for k, v in mdl.config.id2label.items()}
    B = 128 if dev == "cuda" else 32
    print("장치:", dev)
    t0, n_done = time.time(), 0
    for k in todo:
        part = texts[k:k + CHUNK]; S = []
        for i in range(0, len(part), B):
            enc = tok(part[i:i + B], padding=True, truncation=True, max_length=160, return_tensors="pt").to(dev)
            with torch.no_grad():
                p = torch.softmax(mdl(**enc).logits.float(), -1).cpu().numpy()
            S.append(p[:, lab["positive"]] - p[:, lab["negative"]])
        np.save(os.path.join(CK, f"c{k // CHUNK:06d}.npy"), np.concatenate(S).astype("float32"))
        n_done += len(part); sp = n_done / (time.time() - t0)
        left = sum(len(texts[j:j + CHUNK]) for j in todo) - n_done
        print(f"  묶음 {k // CHUNK + 1}/{-(-len(texts) // CHUNK)} 저장 | 초당 {sp:.0f}건 | 남은 시간 약 {left / sp / 60:.0f}분", flush=True)
u["s"] = np.concatenate([np.load(f) for f in sorted(glob.glob(os.path.join(CK, "c*.npy")))])[:len(u)]
x = x.merge(u[["id", "s"]], on="id", how="left")

# 3. 주가 보도 표시, ESG 분류(제목)
x["price_news"] = x.title.fillna("").str.contains(PRICE)
x["esg"] = x.title.fillna("").map(lambda t: classify_esg(t)[0])

# 4. 월 배정 (그달 마지막 거래일 당일·이후 기사는 다음 달) → 회사×월
last = (x.date + pd.offsets.BMonthEnd(0)).dt.normalize(); per = x.date.dt.to_period("M")
x["month"] = per.where(x.date < last, per + 1).astype(str)
x = x[x.month <= "2021-12"]
g = x.groupby(["month", "code"])
mon = pd.DataFrame({"n": g.size(), "s_all": g.s.mean(),
                    "pos_share": g.s.apply(lambda v: (v > 0.5).mean()), "neg_share": g.s.apply(lambda v: (v < -0.5).mean())})
gp = x[~x.price_news].groupby(["month", "code"]); mon["n_noprice"] = gp.size(); mon["s_noprice"] = gp.s.mean()
for lab_, k in [("부정", "esg_neg"), ("긍정", "esg_pos")]:
    sub = x[x.esg == lab_].groupby(["month", "code"]); mon[f"n_{k}"] = sub.size(); mon[f"{k}_tone"] = sub.s.mean()
for c in ["n_noprice", "n_esg_neg", "n_esg_pos"]: mon[c] = mon[c].fillna(0).astype(int)
mon = mon.reset_index()
name = "sent_monthly_test.csv" if TEST else "sent_monthly.csv"
mon.to_csv(os.path.join(OUT, name), index=False, encoding="utf-8-sig")
print(f"끝! {name} {mon.shape} → 이 파일을 Claude에게 보내기 (원문 없음)")
