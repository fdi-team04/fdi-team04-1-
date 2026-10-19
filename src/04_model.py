"""[04] 모형 학습 (train 기간만, 확장 윈도우)   담당: 미정

입력  data/processed/model_input.parquet
출력  data/processed/predictions.parquet  (month, code, target, score, stage)

STAGES (정의서 04): 단계마다 가설 변수를 누적 추가한다.
        열 이름 = 정의서 계산식 기호 → 02·03 출력 열 이름과 같아야 한다.
  0: beta, size, bm     H1 시장 베타, H2 규모, H3 가치 (기준 모형)
  1: + mom, rev         H4 모멘텀, H5 단기 반전
  2: + vol, issue       H6 저변동성, H7 주식 발행
  3: + sent             H8 뉴스 감성 (03)
"""
import pandas as pd
import yaml

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
PROC = f"{cfg['data_dir']}/processed"


def fama_macbeth(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Fama-MacBeth 1단계: 월별 횡단면 회귀

    입력: df — 학습에 쓸 행(train 기간. 마지막 달은 target이 test 첫 달 수익률이라 제외.
          target이 비어 있는 행(다음 달 상장 폐지 등)도 제외).
          cols — 이번 단계 변수.
    출력: DataFrame(index=month, columns=const + cols), 달마다 회귀 계수
    규칙: 매월 target = a + b·cols 를 최소제곱(OLS)으로 추정한다.
    """
    pass


def predict(df: pd.DataFrame, coefs: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """확장 윈도우 예측

    입력: df — 예측할 행(train + test, warmup 제외). coefs — fama_macbeth 출력.
    출력: DataFrame(month, code, target, score)
    시점: t월 점수 = (t월보다 앞선 달들의 계수 평균) · t월 변수.
          t−1월 회귀의 target(= t월 수익률)은 t월 말에 이미 알 수 있으므로 미래정보가 아니다.
          회귀는 train 기간만 하므로 test 달은 train 전체 평균 계수를 쓴다.
    """
    pass


if __name__ == "__main__":
    pass  # 구현 후: STAGES 순서대로 fama_macbeth → predict를 호출해 predictions.parquet 저장
