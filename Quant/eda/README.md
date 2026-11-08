# Quant/eda — 데이터 열어보기 스크립트

`run_all.py`(01→05 공식 파이프라인)와 별개인 탐색용 스크립트다. 결과는 방향 미리 보기이고, 변수 생존 판정은 05의 단계별 IC 표로 한다.

| 파일 | 담당 | 하는 일 |
|---|---|---|
| `target_and_correlation.py` | 위영섭 | 타겟(다음 달 조정수익률) 생성, 월별 종목 수·빈 값·수익률 분포, 규모·단기 반전·모멘텀의 월별 IC와 5분위 수익률 그래프 |
| `pbr_check.py` | 위영섭 | PBR 0·음수 구간, 배포 데이터와의 매칭률 |

```bash
python Quant/eda/target_and_correlation.py <warmup_train_long.csv> --out eda_out/target_corr
python Quant/eda/pbr_check.py <monthly_pbr_2000_2025.csv> --long <warmup_train_long.csv> --out eda_out/pbr
```

- 추가 패키지: `matplotlib`, `tabulate`
- 두 스크립트 모두 2022-01 이후(test) 행은 읽자마자 버린다.
- 결과 폴더 `eda_out/`은 Git에 올리지 않는다.
