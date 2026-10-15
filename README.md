# FDI Team04: KOSPI 다음 달 수익률 순위 예측

금융데이터인텔리전스(2026-2) 팀 프로젝트. 매월 말까지 알 수 있는 정보로 다음 달 종목별 수익률 순위를 예측한다.

## 실행

1. VS Code에서 `notebooks/run_colab.ipynb`를 열고 커널을 Colab에 연결한다.
2. 셀을 위에서부터 실행한다: 데이터 받기 → 코드 받기 → `python run_all.py`

## 구조

| 경로 | 내용 | 담당 |
|---|---|---|
| `config.yaml` | 기간 구분, 데이터 경로, 시드, 텍스트 모델 | 공통 |
| `run_all.py` | `src/01` → `05` 순서 실행 | 공통 |
| `src/01_load_clean.py` | 배포 데이터 결합·시점 정렬 | A |
| `src/02_features.py` | 정형 변수 (H1~H7) | A |
| `src/03_text_features.py` | 텍스트 감성 점수·결합·순위 변환 (H8) | B-2 |
| `src/04_model.py` | 모형 학습 | 미정 |
| `src/05_evaluate.py` | 월별 Spearman IC | 미정 |
| `notebooks/` | Colab 실행, 첫 확인 | 각자 |
| `docs/` | 정의서, 자료 메모, 재수집 방법 | 각자 |

## 규칙

- `data/`는 Git에 올리지 않는다. 배포 데이터는 드라이브 공개 폴더, 빅카인즈 원문은 비공개 폴더에 둔다.
- test(2022~2025)는 학습·전처리·변수 선택에 쓰지 않는다. 기간은 `config.yaml`에서만 정한다.
- `main`에 직접 올리지 않는다. 브랜치 → Pull Request → 1명 승인 후 병합.
- 노트북은 커밋 전에 Clear All Outputs.
