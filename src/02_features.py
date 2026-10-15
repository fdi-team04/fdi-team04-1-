"""[02] 정형 변수 계산 (H1~H7)   담당: A

입력  data/processed/panel.parquet
출력  data/processed/features.parquet  (month, code, 변수들, target = 다음 달 수익률)
"""
import yaml

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
PROC = f"{cfg['data_dir']}/processed"


if __name__ == "__main__":
    pass  # 구현 예정
