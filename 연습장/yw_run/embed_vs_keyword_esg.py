"""ESG 분류: 키워드 규칙(v1.1, v1.2) vs 임베딩 — 사람 정답 100건(ESG_최종정답_100.csv, 부정/긍정/무관)
임베딩: ko-sroberta-multitask, KR-FinBert-SC (제목 전체를 문장 임베딩)
  B. 임베딩 + 학습: 3분류 로지스틱 회귀, 한 건씩 빼고 학습해서 맞히기(leave-one-out)
  C. 임베딩 + 설명문 비교(학습 없음): 부정 ESG / 긍정 ESG / 무관 설명문과의 유사도
추가: 이안 점검에서 나온 '화재' 오류 유형(보험사 이름 vs 실제 화재) 예시 문장으로 각 방법 확인
제목은 저장하지 않음(숫자만). 주의: 키워드 v1.1·v1.2는 이 100건을 보고 고친 규칙이라 참고값
"""
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, "/home/user/fdi-team04-1-/연습장/news")
import esg_rules as v12, esg_rules_v11 as v11
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from transformers import AutoTokenizer, AutoModel
d = pd.read_csv("/home/user/data/esg_in/ESG_최종정답_100.csv"); y = d.최종.str.strip().values
L = ["부정", "긍정", "무관"]
print("정답 분포", pd.Series(y).value_counts().to_dict())
def embed(name, texts):
    tok = AutoTokenizer.from_pretrained(name); m = AutoModel.from_pretrained(name).eval()
    with torch.no_grad():
        e = tok(list(texts), padding=True, truncation=True, max_length=64, return_tensors="pt"); h = m(**e).last_hidden_state
        v = (h * e["attention_mask"][..., None]).sum(1) / e["attention_mask"].sum(1, keepdim=True)
    return torch.nn.functional.normalize(v, dim=1).numpy()
DESC = {"부정": ["검찰이 회사를 압수수색했다", "공장에서 사고로 근로자가 숨졌다", "노조가 파업에 들어갔다", "갑질 논란이 일었다", "오염물질을 불법 배출했다", "횡령 혐의로 기소됐다"],
        "긍정": ["회사가 이웃돕기 성금을 기부했다", "사회공헌 활동을 펼쳤다", "친환경 경영을 강화했다", "협력사와 상생 협약을 맺었다", "윤리경영을 선포했다"],
        "무관": ["분기 실적이 발표됐다", "주가가 올랐다", "신제품을 출시했다", "신임 사장이 취임했다", "목표주가를 올렸다"]}
def rep(name, pred):
    r = {"방법": name, "정확도": (pred == y).mean()}
    for c in ["부정", "긍정"]:
        k = pred == c; r[f"{c} 정밀도"] = (y[k] == c).mean() if k.sum() else np.nan; r[f"{c} 재현율"] = (pred[y == c] == c).mean()
    r["부정↔긍정 혼동"] = int(((pred == "부정") & (y == "긍정")).sum() + ((pred == "긍정") & (y == "부정")).sum()); return r
rows = [rep("키워드 v1.1", np.array([v11.classify_esg(t)[0] for t in d.제목])), rep("키워드 v1.2 (최종)", np.array([v12.classify_esg(t)[0] for t in d.제목]))]
TEST = [("메리츠화재 3분기 순익 증가", "무관"), ("동부화재 사장 취임", "무관"), ("LG화재 자동차보험료 인하", "무관"), ("동양화재컵 골프대회 개막", "무관"),
        ("자산재평가 차익 1조", "무관"), ("공장화재로 3명 사망", "부정"), ("물류센터 화재 책임 공방", "부정"), ("LG화학 공장 폭발", "부정"), ("산재 사망사고 잇따라", "부정")]
tests = {"키워드 v1.1": [v11.classify_esg(t)[0] for t, _ in TEST], "키워드 v1.2 (최종)": [v12.classify_esg(t)[0] for t, _ in TEST]}
for name, lab in [("jhgan/ko-sroberta-multitask", "ko-sroberta"), ("snunlp/KR-FinBert-SC", "KR-FinBert")]:
    X = embed(name, d.제목); pred = np.empty(len(y), dtype=object)
    for tr, te in LeaveOneOut().split(X):
        pred[te] = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(X[tr], y[tr]).predict(X[te])
    rows.append(rep(f"임베딩+학습 ({lab})", pred.astype(str)))
    Dm = np.vstack([embed(name, DESC[c]).mean(0) for c in L]); p0 = np.array(L)[(X @ Dm.T).argmax(1)]
    rows.append(rep(f"임베딩+설명문 ({lab}, 학습 없음)", p0))
    Xt = embed(name, [t for t, _ in TEST]); tests[f"임베딩+설명문 ({lab})"] = list(np.array(L)[(Xt @ Dm.T).argmax(1)])
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(X, y); tests[f"임베딩+학습 ({lab})"] = list(clf.predict(Xt))
R = pd.DataFrame(rows); pd.set_option("display.width", 250); print(R.round(2).to_string(index=False))
T = pd.DataFrame(tests, index=[f"{t} (정답 {g})" for t, g in TEST]); print(T.to_string())
print("오류 예시 9개 정답 수:", {k: int(sum(p == g for p, (_, g) in zip(v, TEST))) for k, v in tests.items()})
R.to_csv("/home/user/fdi-team04-1-/연습장/out_yw/embed_vs_keyword_esg.csv", index=False, encoding="utf-8-sig")
T.to_csv("/home/user/fdi-team04-1-/연습장/out_yw/embed_vs_keyword_esg_화재예시.csv", encoding="utf-8-sig")
