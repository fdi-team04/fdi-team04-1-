"""[03] 텍스트 → 종목·월 점수 → 정형 패널 결합 → 순위 변환   담당: B-1(앞부분) · B-2(점수·결합)

입력  data/processed/articles_clean.csv      (B-1 가공 빅카인즈: code, month, date, title, body)
      data/processed/features.parquet        (A: month, code, 정형 변수, target)

출력  data/processed/article_scores.parquet  (기사별 점수 캐시, 원문 없음)
      data/processed/model_input.parquet     (04_model 입력)

B-1(정아영): load_news, clean_news → articles_clean.parquet 생성
B-2(하이안): assign_month 이후 → model_input.parquet 생성
이용 조건: 빅카인즈 기사 원문은 이용약관상 재배포 제한 → 저장소·제출물에 올리지 않는다.
           수업·학술 목적으로만 쓰고 모형은 추가 학습하지 않는다(두 담당 교수님 회신, 2026-10-18).
"""
import os

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import pipeline

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
PROC = f"{cfg['data_dir']}/processed"
"""[H8] 뉴스 감성 (+, 대립 −): 뉴스 수집·정제 부분

담당: 정아영(B1). 점수 산출(score_sentiment, news_sentiment)은 하이안(B2)이 이어서 작성한다.

이론: 기업 관련 기사에 담긴 정보는 이후 실적을 예측하지만 주가가 즉시 다 반영하지 못한다
      (Tetlock, Saar-Tsechansky & Macskassy, 2008). 대립 가설은 비관적 논조 뒤 되돌림(Tetlock, 2007).
자료: D5 빅카인즈 경제지 5곳(머니투데이, 매일경제, 파이낸셜뉴스, 서울경제, 한국경제), 연 단위 엑셀.
이용 조건: 기사 원문은 이용약관상 재배포 제한 → 저장소와 제출물에 올리지 않는다(data/raw는 .gitignore).
           수업·학술 목적으로만 쓰고 모형은 추가 학습하지 않는다(두 담당 교수님 회신, 2026-10-18).
"""


def load_news(raw_dir, codes):
    """[H8] 뉴스 원자료 적재 (+, 대립 −)

    이론: 위 모듈 설명 참조.
    입력: raw_dir — data/raw/news/ 아래 빅카인즈 엑셀(연 단위 파일).
          codes — 대상 종목코드 목록(6자리 문자열, 앞자리 0 유지).
    출력: DataFrame(date, outlet, title, body, excluded_flag, source_file)
    규칙: 일자는 YYYYMMDD(시각 없음). 본문은 200자 이내만 제공된다.
    시점: t월 말까지 공개된 기사만 이후 단계에서 쓴다(여기서는 적재만).
    """
    pass


def clean_news(df, name_dict):
    """[H8] 분석제외·중복 제거와 제목 기준 종목 매칭 (+, 대립 −)

    입력: df — load_news 출력.
          name_dict — {종목코드: [시점별 회사명]} (배포 자료의 company_name에서 만든다).
    출력: DataFrame(code, date, title, body)
    규칙: ① 분석제외 표시('중복', '예외')가 있는 기사 제거
          ② (제목, 언론사, 일자)가 같은 기사 제거
          ③ 제목에 회사명이 있는 기사만 남김
          ④ 기관 열에만 회사명이 있는 기사는 보류(사용하지 않음)
    기사 수는 신뢰도 점검용으로만 기록하고 예측 변수로 쓰지 않는다.
    """
    pass


def assign_month(df, trading_days):
    """[H8] 월 귀속 (+, 대립 −)

    입력: df — clean_news 출력.
          trading_days — 거래일 목록(월별 마지막 거래일 판별용).
    출력: DataFrame(code, month, date, title, body)
    규칙: t월 마지막 거래일 전날까지의 기사는 t월, 마지막 거래일 이후 기사는 t+1월로 보낸다.
    시점: 시각이 없으므로 보수적으로 마지막 거래일 당일 기사는 t+1월로 보낸다(미래정보 차단).
    기사가 없는 달은 결측으로 두고 2.3절 규칙(월별 중앙값 + 결측 표시 열)을 따른다.
    """
    pass


# ---- 아래는 B2(하이안) 담당: score_articles, aggregate_monthly, merge_text_features, rank_transform ----

def finbert_probs(texts: pd.Series) -> pd.DataFrame:
    """[H8] KR-FinBert-SC 라벨별 확률 (+, 대립 −)

    입력: texts — 기사 텍스트(제목 + 본문).
    출력: DataFrame(negative, neutral, positive), texts와 같은 인덱스
    규칙: 사전학습 모델을 그대로 쓰고 재학습하지 않는다(이용 조건).
          config.yaml에 text_model_rev(커밋 해시)를 적으면 그 버전으로 고정된다.
    """
    clf = pipeline("text-classification", model=cfg["text_model"],
                   revision=cfg.get("text_model_rev"), top_k=None, truncation=True,
                   device=0 if torch.cuda.is_available() else -1)
    out = clf(texts.tolist(), batch_size=32)
    return pd.DataFrame([{d["label"]: d["score"] for d in o} for o in out], index=texts.index)


def score_articles(articles: pd.DataFrame, backend: str = "finbert") -> pd.DataFrame:
    """[H8] 기사별 감성 점수 (+, 대립 −)

    입력: articles — B-1 가공 빅카인즈(code, month, date, title, body).
          backend — "finbert"(config의 text_model) 또는 "llm"(모델 확정 후 추가).
    출력: DataFrame(code, month, date, s), s = P(긍정) − P(부정) ∈ [−1, 1]
    규칙: 제목 + 본문(200자)을 넣는다. 모델 버전 고정, 재학습 없음.
          결과는 article_scores.parquet에 캐시해 재사용한다(원문 열은 저장하지 않음).
          입력 CSV나 backend를 바꾸면 캐시 파일을 지우고 다시 돌린다.
    """
    path = f"{PROC}/article_scores.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)

    texts = articles["title"] + " " + articles["body"].fillna("")
    if backend == "finbert":
        p = finbert_probs(texts)
    else:
        raise NotImplementedError("llm은 모델 확정 후 finbert_probs와 같은 형식으로 추가")

    scored = articles.drop(columns=["title", "body"]).assign(s=p["positive"] - p["negative"])
    scored.to_parquet(path, index=False)
    return scored


def aggregate_monthly(scored: pd.DataFrame) -> pd.DataFrame:
    """[H8] 종목·월 감성 지수 (+, 대립 −)

    입력: scored — score_articles 출력.
    출력: DataFrame(month, code, sent, news_cnt)
    규칙: sent = 그 달 기사 s의 평균, news_cnt = log(1 + 기사 수).
          기사가 없는 (month, code)는 행이 없으므로 결합 후 결측이 된다.
    """
    g = scored.groupby(["month", "code"])["s"]
    return pd.DataFrame({"sent": g.mean(), "news_cnt": np.log1p(g.size())}).reset_index()


def merge_text_features(panel: pd.DataFrame, text: pd.DataFrame) -> pd.DataFrame:
    """[H8] 정형 패널에 텍스트 변수 결합 (+, 대립 −)

    입력: panel — features.parquet(month, code, 정형 변수, target).
          text — aggregate_monthly 출력.
    출력: panel + (sent, news_cnt, in_text_sample)
    규칙: left join (키: month, code / validate="one_to_one"), 결합 전후 행 수가 같은지 확인.
          in_text_sample = 뉴스 수집 대상 종목(text에 한 번이라도 나온 code)이면 1.
          수집 대상인데 그 달 기사가 0건이면 news_cnt = 0(= log 1), sent는 결측으로 둔다.
    """
    out = panel.merge(text, on=["month", "code"], how="left", validate="one_to_one")
    assert len(out) == len(panel), "결합 전후 행 수가 다르다"

    sample = out["code"].isin(text["code"])
    out["in_text_sample"] = sample.astype(int)
    out.loc[sample, "news_cnt"] = out.loc[sample, "news_cnt"].fillna(0)
    return out


def rank_transform(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """월별 횡단면 순위 변환 (2.3절 규칙)

    입력: panel — merge_text_features 출력. cols — 변환할 변수 목록.
    출력: panel(cols를 0~1 순위 백분위로 교체) + 변수명_missing(0/1)
    규칙: 순위 백분위 = rank / (n + 1) (n = 그 달 값이 있는 종목 수, 동순위는 평균 순위). 정의서 공통 규칙.
          결측 = 0.5(중앙값) + 변수명_missing. 결측이 없는 변수도 표시 열을 만들어 기간별 열 구성을 같게 한다.
    시점: 그 달 단면만 쓰므로 기간을 넘는 통계 추정이 없다(test 누수 방지).
    """
    out = panel.copy()
    for c in cols:
        out[f"{c}_missing"] = out[c].isna().astype(int)
        g = out.groupby("month")[c]
        out[c] = (g.rank() / (g.transform("count") + 1)).fillna(0.5)
    return out


if __name__ == "__main__":
    news = pd.read_csv(f"{PROC}/articles_clean.csv", dtype={"code": str})  # 종목코드 앞자리 0 유지
    text = aggregate_monthly(score_articles(news))
    panel = merge_text_features(pd.read_parquet(f"{PROC}/features.parquet"), text)

    feats = [c for c in panel.select_dtypes("number") if c not in ("target", "in_text_sample")]
    rank_transform(panel, feats).to_parquet(f"{PROC}/model_input.parquet", index=False)
