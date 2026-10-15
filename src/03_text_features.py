"""[03] 텍스트 → 종목·월 점수 → 정형 패널 결합 → 순위 변환   담당: B-2

입력  data/processed/articles_clean.parquet  (B-1: date, title, body, source, company_name)
      data/processed/features.parquet        (A: month, code, 정형 변수, target)
출력  data/processed/article_scores.parquet  (기사별 점수 캐시, 원문 없음)
      data/processed/model_input.parquet     (04_model 입력)
"""
import pandas as pd
import yaml

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
PROC = f"{cfg['data_dir']}/processed"


def assign_month(articles: pd.DataFrame, last_trading_day: pd.Series) -> pd.DataFrame:
    """기사 날짜 → 예측 월(month) 배정.
    t월 창 = [t-1월 마지막 거래일, t월 마지막 거래일).
    일자만 있고 시각이 없으므로 마지막 거래일 당일·이후 기사는 t+1월로 보낸다."""
    pass


def match_codes(articles: pd.DataFrame, aliases: pd.DataFrame) -> pd.DataFrame:
    """제목의 회사명 → 6자리 종목코드(code).
    aliases = (code, name, start, end) 옛 이름·약칭 사전. 이름 완전 일치만 채택,
    한 제목에 여러 종목이면 행 복제, 매칭 실패는 따로 기록."""
    pass


def score_articles(articles: pd.DataFrame, backend: str = "finbert") -> pd.DataFrame:
    """[H8] 뉴스 감성 · 방향 +
    기사별 s = P(긍정) - P(부정). backend = "finbert"(config의 text_model) 또는 "llm".
    모델 버전 고정, 재학습 없음. 결과는 article_scores.parquet에 캐시해 재사용."""
    pass


def aggregate_monthly(scored: pd.DataFrame) -> pd.DataFrame:
    """기사 점수 → (month, code)별 sent = s 평균, news_cnt = log(1 + 기사 수)."""
    pass


def merge_text_features(panel: pd.DataFrame, text: pd.DataFrame) -> pd.DataFrame:
    """정형 패널에 left join (키: month, code / validate="one_to_one").
    결합 전후 행 수가 같은지 확인하고, 뉴스 수집 대상 표시 in_text_sample을 붙인다."""
    pass


def rank_transform(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """월별 횡단면 순위를 [-1, 1]로 변환, 결측 = 0(중앙값) + 변수명_missing.
    그 달 단면만 쓰므로 기간을 넘는 통계 추정이 없다 (test 누수 방지)."""
    pass


if __name__ == "__main__":
    pass  # 구현 후: 위 함수를 순서대로 호출해 model_input.parquet 저장
