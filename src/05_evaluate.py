"""[05] 평가: 월별 Spearman IC   담당: 미정

입력  data/processed/predictions.parquet
출력  data/processed/ic_monthly.csv
"""
import yaml

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
PROC = f"{cfg['data_dir']}/processed"


if __name__ == "__main__":
    pass  # 구현 예정
