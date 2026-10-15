"""[04] 모형 학습 (train 기간만, 확장 윈도우)   담당: 미정

입력  data/processed/model_input.parquet
출력  data/processed/predictions.parquet  (month, code, score)
"""
import yaml

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
PROC = f"{cfg['data_dir']}/processed"


if __name__ == "__main__":
    pass  # 구현 예정
