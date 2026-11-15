"""2-4 감성 점수: KR-FinBert-SC로 기사마다 s = P(긍정) − P(부정) 계산 → 월 배정 → 회사×월 집계

쓰는 법 (Colab, 런타임 T4 GPU 권장):
    !pip install transformers
    exec(open('2-4_감성점수.py', encoding='utf-8').read())   # 이 파일 불러오기
    fb = load_finbert()                    # 모델 받기 (처음 1회 1~2분)
    x['s'] = score((x.title + ' ' + x.body.str[:200]).tolist(), fb)   # 광고는 빼고 넣기
    x['month'] = assign_month(x.date)      # 마지막 거래일 당일·이후 기사는 다음 달로
    mon = monthly(x)                       # 회사×월 표 (원문 없음 → 공유해도 됨)
입력 x 칸: id, code, date(datetime), title, body, type(2-3 분류 결과)
재학습 없음. 제목 + 본문 앞 200자만 씀.
"""
import numpy as np, pandas as pd

# ---------- 2-4 감성 점수 ----------
def load_finbert(model="snunlp/KR-FinBert-SC"):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModelForSequenceClassification.from_pretrained(model).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; mdl.to(dev)
    if dev == "cuda": mdl.half()
    lab = {v.lower(): k for k, v in mdl.config.id2label.items()}
    print("모델", model, "| 장치", dev, "| 라벨", mdl.config.id2label)
    return tok, mdl, dev, lab


def score(texts, fb, bs=128, max_len=160):
    """s = P(긍정) − P(부정), −1~+1. 재학습 없음"""
    import torch
    tok, mdl, dev, lab = fb; out = []
    for i in range(0, len(texts), bs):
        enc = tok(list(texts[i:i + bs]), padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
        with torch.no_grad():
            p = torch.softmax(mdl(**enc).logits.float(), -1).cpu().numpy()
        out.append(p[:, lab["positive"]] - p[:, lab["negative"]])
    return np.concatenate(out) if out else np.array([])


def assign_month(dates):
    """마지막 거래일(평일 근사) 당일·이후 기사는 다음 달로"""
    per = dates.dt.to_period("M")
    last = (dates + pd.offsets.BMonthEnd(0)).dt.normalize()
    return per.where(dates < last, per + 1).astype(str)


def monthly(x):
    """회사×월 집계 (원문 없음 → 밖으로 내보내도 됨)"""
    g = x.groupby(["month", "code"])
    out = g.size().rename("n_all").to_frame()
    for t, k in [("화제형", "topic"), ("공시형", "disc"), ("논란형", "contro"), ("광고", "ad")]:
        sub = x[x.type == t].groupby(["month", "code"])
        out[f"n_{k}"] = sub.size(); out[f"{k}_tone"] = sub.s.mean()
    for c in [c for c in out if c.startswith("n_")]:
        out[c] = out[c].fillna(0).astype(int)
    return out.reset_index()
