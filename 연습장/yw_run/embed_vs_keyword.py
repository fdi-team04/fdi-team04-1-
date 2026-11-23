"""키워드 vs 임베딩 분류 비교 (교수님 11/18 질문) — 아영·예원 50건 판정을 정답으로 사용
정답: 두 사람 답이 같은 기사만 (실적 35건: 맞음 27·아님 8 / 목표가 50건: 합의 후 전부 맞음)
비교 방법
  A. 키워드 v0: 키워드에 걸린 기사 = 전부 '맞음'으로 분류 (실제로 쓴 방식)
  B. 임베딩 + 분류기: 제목을 문장 임베딩으로 바꾸고 로지스틱 회귀, 한 건씩 빼고 학습해서 맞히기(leave-one-out) → 학습에 쓴 기사로 채점하지 않음
  C. 임베딩 + 설명문 비교(학습 없음): 제목과 '이미 나온 좋은 실적' / '앞으로의 전망' / '실적과 무관' 설명문의 유사도로 분류
  임베딩 모델 2개: ko-sroberta-multitask(한국어 문장 임베딩), KR-FinBert-SC(금융 감성 모델의 내부 표현)
기사 제목은 결과 파일에 저장하지 않음 (숫자만)
쓰는 법: python embed_vs_keyword.py <실적 판정A.xlsx> <실적 판정B.xlsx> <목표가 판정A.xlsx>
"""
import sys, numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from transformers import AutoTokenizer, AutoModel
col = "사람 판정(맞음/아님/애매)"
a = pd.read_excel(sys.argv[1]); b = pd.read_excel(sys.argv[2]); tp = pd.read_excel(sys.argv[3])
A, B = a[col].str.strip(), b[col].str.strip()
gold = a[(A == B) & A.isin(["맞음", "아님"])].copy(); gold["y"] = (A[gold.index] == "맞음").astype(int)
print(f"실적 정답 {len(gold)}건 (맞음 {gold.y.sum()}, 아님 {len(gold)-gold.y.sum()})")
def embed(name, texts):
    tok = AutoTokenizer.from_pretrained(name); m = AutoModel.from_pretrained(name).eval()
    with torch.no_grad():
        e = tok(list(texts), padding=True, truncation=True, max_length=64, return_tensors="pt"); h = m(**e).last_hidden_state
        v = (h * e["attention_mask"][..., None]).sum(1) / e["attention_mask"].sum(1, keepdim=True)
    return torch.nn.functional.normalize(v, dim=1).numpy()
DESC = {"결과": ["회사의 분기 실적이 좋게 발표됐다", "영업이익이 사상 최대를 기록했다", "흑자로 전환했다"],
        "전망": ["실적이 앞으로 좋아질 것으로 예상된다", "실적 개선이 지속될 전망이다", "흑자 전환을 목표로 한다"],
        "무관": ["주가가 크게 올랐다", "생산량이 늘었다", "과징금이 부과됐다"]}
def score(y, pred):
    kept = pred == 1; prec = y[kept].mean() if kept.sum() else np.nan
    return {"정확도": (pred == y).mean(), "남긴 기사 수": int(kept.sum()), "정밀도(남긴 것 중 맞음)": prec, "진짜 호재를 놓친 수": int(((pred == 0) & (y == 1)).sum())}
rows = [{"방법": "A. 키워드 v0 (실제 사용)", **score(gold.y.values, np.ones(len(gold), int))}]
for name, lab in [("jhgan/ko-sroberta-multitask", "ko-sroberta"), ("snunlp/KR-FinBert-SC", "KR-FinBert")]:
    X = embed(name, gold.제목); y = gold.y.values; pred = np.zeros(len(y), int)
    for tr, te in LeaveOneOut().split(X):
        pred[te] = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(X[tr], y[tr]).predict(X[te])
    rows.append({"방법": f"B. 임베딩+분류기 ({lab})", **score(y, pred)})
    D = {k: embed(name, v).mean(0) for k, v in DESC.items()}
    sims = np.column_stack([X @ D[k] for k in ["결과", "전망", "무관"]]); pred0 = (sims.argmax(1) == 0).astype(int)
    rows.append({"방법": f"C. 임베딩+설명문 ({lab}, 학습 없음)", **score(y, pred0)})
    if lab == "ko-sroberta":
        Xt = embed(name, tp.제목); D2 = embed(name, ["증권사가 목표주가를 올렸다", "투자의견을 상향했다"]).mean(0); D3 = embed(name, ["주가가 크게 올랐다", "실적이 발표됐다"]).mean(0)
        tp_keep = int(((Xt @ D2) > (Xt @ D3)).sum())
R = pd.DataFrame(rows); pd.set_option("display.width", 250); print(R.round(3).to_string(index=False))
print(f"목표가 50건(전부 맞음): 키워드 정밀도 100% / 임베딩 설명문 방식(ko-sroberta)이 '목표가 상향'으로 남긴 수 {tp_keep}/50")
R.to_csv("/home/user/fdi-team04-1-/연습장/out_yw/embed_vs_keyword.csv", index=False, encoding="utf-8-sig")
