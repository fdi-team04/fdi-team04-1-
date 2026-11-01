"""[05] 평가: 월별 Spearman IC   담당: 정아영

입력  data/processed/predictions.parquet  (04: month, code, target, score, stage)
출력  data/processed/ic_monthly.csv       (month, stage, ic, n_stocks)
      data/processed/ic_summary.csv       (stage, mean_ic, pos_ratio, min_ic, n_months)

평가 규칙(공지): 매월 "그 달 실제 수익률(target)이 있는 모든 종목"에서 예측값(score)과
실제 수익률의 Spearman 순위상관을 계산한다. 단계(stage)별로 따로 계산한다.
train 검증 구간은 2011~2021년(확장 윈도우, 11개 검증 연도), test 48개월은 --final에서만 쓴다.
"""
import pandas as pd
import yaml

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
PROC = f"{cfg['data_dir']}/processed"


def load_predictions(path: str) -> pd.DataFrame:
    """[공통] 예측 파일 적재

    입력: path — predictions.parquet 경로.
    출력: DataFrame(month, code, target, score, stage), month = "YYYY-MM" 문자열, code = 6자리 문자열
    규칙: target이 비어 있는 행(다음 달 수익률 없음)은 평가에서 제외한다.
    """
    pass


def monthly_ic(pred: pd.DataFrame) -> pd.DataFrame:
    """[공통] 월별 Spearman IC (정보계수)

    입력: pred — load_predictions 출력.
    출력: DataFrame(month, stage, ic, n_stocks)
    규칙: 매월·단계별로 score와 target의 Spearman 순위상관을 계산한다(동순위는 평균 순위).
          종목이 너무 적은 달(예: 10개 미만)은 ic를 결측으로 두고 n_stocks에 개수를 남긴다.
    시점: score는 t월 말 정보로 만든 값이고 target은 t+1월 수익률이다(미래정보 아님).
    """
    pass


def ic_summary(ic: pd.DataFrame) -> pd.DataFrame:
    """[공통] 단계별 IC 요약

    입력: ic — monthly_ic 출력.
    출력: DataFrame(stage, mean_ic, pos_ratio, min_ic, n_months)
    규칙: 평균 IC, IC가 양(+)인 달의 비율, 가장 낮은 IC, 달 수를 계산한다.
          정의서 3.1의 판정(평균 IC 상승)에 쓰이며 최저 IC는 급반등 국면 점검에 쓴다.
    """
    pass


def ic_by_year(ic: pd.DataFrame) -> pd.DataFrame:
    """[공통] 연도별 평균 IC

    입력: ic — monthly_ic 출력.
    출력: DataFrame(year, stage, mean_ic)
    규칙: 11개 검증 연도(2011~2021) 가운데 6개 이상에서 직전 단계보다 연평균 IC가
          높아져야 추가 변수를 유지한다(보고서 3.1 조건 3). 이 표가 그 판정의 입력이다.
    """
    pass


def save_outputs(ic: pd.DataFrame, summary: pd.DataFrame, out_dir: str) -> None:
    """[공통] 결과 저장

    입력: ic, summary — 위 함수 출력. out_dir — data/processed.
    출력: ic_monthly.csv, ic_summary.csv (UTF-8, 인덱스 없음).
    """
    pass


if __name__ == "__main__":
    pass  # 구현 후: load_predictions → monthly_ic → ic_summary → save_outputs 순서로 호출
