"""[03] 텍스트 → 종목·월 점수 → 정형 패널 결합 → 순위 변환   담당: B-1(앞부분) · B-2(점수·결합)

입력  data/processed/articles_clean.parquet  (B-1: date, title, body, source, company_name)
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

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
PROC = f"{cfg['data_dir']}/processed"


# ===== B-1(정아영): 뉴스 수집·정제 → articles_clean.parquet =====

def load_news(raw_dir: str) -> pd.DataFrame:
    """[H8] 빅카인즈 엑셀(연 단위, 경제지 5곳) 적재.
    입력: raw_dir — 비공개 폴더의 엑셀 파일들(저장소에 올리지 않음).
    출력: date, title, body(200자 이내), source, company_name(검색어), excluded_flag.
    일자는 YYYYMMDD(시각 없음)."""
    pass


def clean_news(df: pd.DataFrame) -> pd.DataFrame:
    """[H8] 기사 정제 → data/processed/articles_clean.parquet 저장.
    ① 분석제외('중복','예외') 제거 ② (제목, 언론사, 일자) 중복 제거
    ③ 제목에 company_name이 있는 기사만 남김 ④ 기관 열에만 언급된 기사는 제외.
    출력: date, title, body, source, company_name (원문은 저장소에 올리지 않음)."""
    pass


# ===== B-2(하이안): 월 배정 · 종목 매칭 · 감성 점수 · 결합 · 순위 변환 =====

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
