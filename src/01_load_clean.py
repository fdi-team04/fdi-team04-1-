"""[01] 배포 데이터 불러오기·결합·시점 정렬   담당: A

입력  data/raw/  (warmup·train CSV, 행=월 열=6자리 종목코드)
출력  data/processed/panel.parquet  (long 형식: month, code, ret, me, ...)
"""
import yaml

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
PROC = f"{cfg['data_dir']}/processed"


if __name__ == "__main__":
    pass  # 구현 예정
