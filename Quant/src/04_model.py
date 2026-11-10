"""[04] 확장 윈도우 학습

입력  data/processed/model_input.parquet
출력  data/processed/predictions.parquet  (month, code, target, score, stage)
      stage = "xgb/0"처럼 모델/단계로 적는다 → 05를 고치지 않아도 그대로 돌아간다.

STAGES: 단계마다 변수를 누적해 더하고, 같은 조건으로 xgb(메인)와 ridge(선형 기준선)를 돌린다.
        열 이름은 02·03 출력 열 이름과 같아야 한다.
"""
import pandas as pd
import yaml

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
PROC = f"{cfg['data_dir']}/processed"

STAGES = [
    ["beta", "size", "bm"],            # 0 기준선
    ["mom", "rev"],                    # 1 통제
    ["vol", "issue"],                  # 2 유지 후보
    ["retail_nb", "foreign_nb"],       # 3 수급 (H1, H2)
    ["topic_tone"],                    # 4 뉴스 (H3)
]


def make_model(name: str):
    """'xgb' → XGBRegressor(**cfg['xgb'], random_state=cfg['seed']), 'ridge' → Ridge"""
    pass


def expanding_predict(df: pd.DataFrame, cols: list[str], name: str) -> pd.DataFrame:
    """확장 윈도우 예측

    입력: df — model_input 전체. cols — 이번 단계까지 누적된 변수. name — 'xgb' 또는 'ridge'.
    출력: DataFrame(month, code, target, score)
    시점: valid 시작(cfg['periods']['valid'][0])부터 매달, t월보다 앞선 달로만 학습해 t월을 예측한다.
          t−1월 행의 target(= t월 수익률)은 t월 말에 이미 알 수 있으므로 미래정보가 아니다.
          test 구간은 cfg['test_refit'] 주기로만 재학습한다.
    """
    pass


if __name__ == "__main__":
    pass  # STAGES를 누적하며 xgb·ridge 각각 expanding_predict → 하나로 합쳐 predictions.parquet 저장
