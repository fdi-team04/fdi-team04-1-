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
import pandas as pd
import yaml
import os

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
PROC = f"{cfg['data_dir']}/processed"


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


def classify_type(df):           # B-1 쪽 assign_month 다음에 추가 (뉴스 계획상 정아영 담당)
    """news_type = 공시형 / 화제형 / 논란형. 키워드가 겹치면 공시형 우선"""

def finbert_probs(texts: pd.Series) -> pd.DataFrame:
    """[H8] KR-FinBert-SC 라벨별 확률 (+, 대립 −)

    입력: texts — 기사 텍스트(제목 + 본문).
    출력: DataFrame(negative, neutral, positive), texts와 같은 인덱스
    규칙: 사전학습 모델을 그대로 쓰고 재학습하지 않는다(이용 조건).
          config.yaml에 text_model_rev(커밋 해시)를 적으면 그 버전으로 고정된다.
    """
    from transformers import pipeline
    clf = pipeline("text-classification", model=cfg["text_model"], revision=cfg.get("text_model_rev"), top_k=None, truncation=True) # 토큰제한 512 안전장치
    out = clf(texts.tolist(), batch_size=32) # 32개씩 묶어계산
    probs = pd.DataFrame([{d["label"]: d["score"] for d in o} for o in out], index=texts.index)
    return probs[["negative", "neutral", "positive"]]


def score_articles(articles: pd.DataFrame, backend: str = "finbert") -> pd.DataFrame:
    """[H8] 기사별 감성 점수 (+, 대립 −)

    입력: articles — B-1 가공 빅카인즈(code, month, date, title, body).
          backend — "kr finbert sc"(config의 text_model) + "llm"(upstage ai).
    출력: DataFrame(code, month, date, s), s = P(긍정) − P(부정) ∈ [−1, 1]
    규칙: 제목 + 본문(200자)을 넣는다. 모델 버전 고정, 재학습 없음.
          결과는 article_scores.parquet에 캐시해 재사용한다(원문 열은 저장하지 않음).
          입력 CSV나 backend를 바꾸면 캐시 파일을 지우고 다시 돌린다.
    """
    if backend == "finbert":
        raise NotImplementedError("LLM(Upstage) 백엔드는 나중에 추가")
    cache = f"{PROC}/article_scores.parquet" # 캐시가 있으면 재계산하지 않고 바로 읽음
    if os.path.exists(cache):
        return pd.read_parquet(cache)

    text = articles["title"] + " " + articles["body"]
    uniq = text.drop_duplicates()
    print(f"[score_articles] 기사 {len(text)}건 중 고유 텍스트 {len(uniq)}건 감성점수 계산")
    p = finbert_probs(uniq)
    s = (p["positive"] - p["negative"]).set_axis(uniq)

out = articles.drop(columns= ["title", "body"]).assign(s=text.map(s))
out.to_parquet(cache, index=False)
return out

def aggregate_monthly(scored: pd.DataFrame) -> pd.DataFrame:
    """[H8] 종목·월 감성 지수 (+, 대립 −)

    입력: scored — score_articles 출력.
    출력: DataFrame(month, code, topic_tone, n_topic)
    규칙: 화제형(news_type == "화제형") 기사만 쓴다.
          topic_tone = 그 달 화제형 기사 s의 평균, n_topic = 화제형 기사 수(점검용, 모델에 넣지 않음).
          기사가 없는 (month, code)는 행이 없으므로 결합 후 결측이 된다.
    """
    topic = scored[scored["news_type"] == "화제형"]
    return (topic.groupby(["month", "code"], as_index=False) # 같은 달,종목 기사 끼리 묶음
                 .agg(topic_tone=("s", "mean"), n_topic=("s", "size"))) # 기사 수

def merge_text_features(panel: pd.DataFrame, text: pd.DataFrame) -> pd.DataFrame:
    """[H8] 정형 패널에 텍스트 변수 결합 (+, 대립 −)

    입력: panel — features.parquet(month, code, 정형 변수, target).
          text — aggregate_monthly 출력.
    출력: panel + (topic_tone, n_topic, no_news)
    규칙: left join (키: month, code / validate="one_to_one"), 결합 전후 행 수가 같은지 확인.
          no_news = 그 달 화제형 기사가 없으면 1 (수집 대상 밖 종목 포함). 이때 n_topic = 0,
          topic_tone은 결측으로 둔다.
    """
    out = panel.merge(text, how="left", on=["month", "code"], validate="one_to_one") # 표 붙이기
    assert len(out) == len(panel), "결합 후 행 수가 바뀜"
    out["no_news"] = out["n_topic"].isna().astype(int)      # 기사없음
    out["n_topic"] = out["n_topic"].fillna(0).astype(int)   # 빈칸정리
    return out

def rank_transform(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """월별 횡단면 순위 변환 (2.3절 규칙)

    입력: panel — merge_text_features 출력. cols — 변환할 변수 목록(target·n_topic·no_news는 제외해 원값 유지).
    출력: panel(cols를 0~1 순위 백분위로 교체) + 변수명_missing(0/1)
    규칙: 순위 백분위 = rank / (n + 1) (n = 그 달 값이 있는 종목 수, 동순위는 평균 순위). 정의서 공통 규칙.
          결측 = 0.5(중앙값) + 변수명_missing. 결측이 없는 변수도 표시 열을 만들어 기간별 열 구성을 같게 한다.
    시점: 그 달 단면만 쓰므로 기간을 넘는 통계 추정이 없다(test 누수 방지).
    """
    g = panel.groupby("month")[cols] # 월별로 묶어 순위 변환
    pct = g.rank() / (g.transform("count") + 1) # 동순위 평균, 결측은 제외
    out = panel.copy()
    out[cols] = pct.fillna(0.5) # 결측은 중앙값 0.5로 채움
    return out.join(miss)


if __name__ == "__main__":
    pass  # 구현 후: 위 함수를 순서대로 호출해 model_input.parquet 저장