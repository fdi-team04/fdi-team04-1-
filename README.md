# FDI Team04: KOSPI 다음 달 수익률 순위 예측

금융데이터인텔리전스(2026-2) 팀 프로젝트. 매월 말까지 알 수 있는 정보로 다음 달 종목별 수익률 순위를 예측한다.

## 실행 (Docker)

Docker Desktop을 켠 상태에서, 모든 명령은 저장소 맨 위 폴더(`docker-compose.yml`이 있는 곳)에서 입력한다.

### 빌드

```bash
docker compose up --build
```

처음에는 Python 패키지를 받느라 몇 분 걸린다. 마지막에 `ERROR` 없이 끝나면 성공이다.

### 확인

```bash
docker compose up --build
```

### 나가기
```bash
ctrl + C
```

`Python 3.11.x`가 나오고, `▶ src/04_model.py`부터 `▶ src/06_explain.py`까지 출력되면 준비 끝이다.

### 평소 사용

1. `git pull`
2. `Quant/requirements.txt`나 `Quant/Dockerfile`이 바뀌었으면 `docker compose build`
3. `docker compose run --rm pipeline`

| 바꾼 것 | 다시 빌드? |
|---|---|
| `Quant/src/`, `Quant/config.yaml`, `Quant/run_all.py` | 아니오. 바로 실행한다 (폴더가 연결돼 있음) |
| `Quant/requirements.txt`, `Quant/Dockerfile` | 예. `docker compose build` |

- 성공 여부: 실행 직후 `echo $?`(Windows PowerShell은 `$LASTEXITCODE`)가 `0`이면 성공. 실패하면 `Traceback`이 나오고, 바로 위 `▶` 줄이 실패한 단계다.
- 결과는 내 컴퓨터의 `Quant/data/processed/`에 바로 생긴다.
- Docker는 **CPU만** 쓴다. `run_all.py`는 기본으로 04 → 06만 돌린다 (만들어진 입력 파일 → 결과, 채점용 재현). `python run_all.py --all`이면 01 → 06 전부 돌린다 (원자료·외부 데이터·기사가 있을 때, 아래 "데이터 만들기" 참고).
- test까지 예측하려면 `docker compose run --rm pipeline python run_all.py --final` (검증은 `--final` 없이).
- 코드 작성용 로컬 Python(conda 등)은 각자 자유다. 버전은 3.11로 맞추고, **보고서에 쓰는 결과는 Docker로 낸다.**
- Windows에서 실행이 느리면 저장소를 WSL 안(`\\wsl$\Ubuntu\home\...`)에 받으면 빨라진다.

## 구조

```
Team04/
├── docker-compose.yml
├── README.md
├── docs/
└── Quant/
    ├── Dockerfile  requirements.txt  config.yaml  run_all.py
    ├── src/
    ├── tables/      (저장소에 올리는 작은 표, 원문 없음)
    └── data/        (Git 제외)
```

| 경로 | 내용 | 담당 |
|---|---|---|
| `docker-compose.yml` | Docker 실행 설정: 폴더 연결, `.env`, 실행 명령 | 공통 |
| `Quant/Dockerfile` | Docker 이미지: Python 3.11 + 패키지 설치 | 공통 |
| `Quant/requirements.txt` | 패키지 버전 고정 | 공통 |
| `Quant/config.yaml` | 기간 구분, 데이터 경로(기사·모델 입력), 시드, XGBoost 설정 | 공통 |
| `Quant/run_all.py` | 기본 `src/04` → `06` (만들어진 데이터 → 결과), `--all`이면 `01` → `06` | 공통 |
| `Quant/src/01_load_clean.py` | 배포 데이터 결합·시점 정렬 | A |
| `Quant/src/02_features.py` | 정형 변수 (기준선·통제·유지 후보) | A |
| `Quant/src/03_text_features.py` | 정리된 기사 → 목표가 상향 종목·월 표 (`tables/tp_monthly.csv`) | B-2 |
| `Quant/src/match_rules.py` | 제목 → 종목 매칭 규칙 v2 (기사 정리 코드가 불러 쓴다, 증권사 꼬리표 제거 포함) | B-1 |
| `Quant/src/make_trading_days.py` | 월별 마지막 거래일 받기 (한 번만, Docker 밖에서) | B-2 |
| `Quant/src/04_model.py` | 모델 입력 준비(월별 순위·결측 표시) + 확장 윈도우 학습·예측 | 미정 |
| `Quant/src/05_evaluate.py` | 월별 Spearman IC, 단계 판정 (전체·상위 200) | 미정 |
| `Quant/src/06_explain.py` | 방향 확인과 해석 (Fama-MacBeth: 정형 9개·H3·H4·규모별, SHAP) | 미정 |
| `Quant/tables/` | 목표가 상향 종목·월 표, 마지막 거래일 (원문 없음, Git에 올림) | B-2 |
| `Quant/data/` | `raw/` 원자료, `processed/` 결과 (Git 제외, 각자 드라이브에서 받음) | 각자 |
| `docs/` | 정의서, 자료 메모, 재수집 방법 | 각자 |

## 데이터 만들기 (01–03)

원자료·기사 원문이 있는 곳에서 `python run_all.py --all`(또는 단계별 실행)로 파일을 만든다. 기본 실행은 이 파일로 04 → 06만 돌린다.

- **정형 (01·02, 합치기)**: 배포 데이터 + 외부 데이터 → 정형 변수. 합치기 코드가 `tables/tp_monthly.csv`를 붙여 `data/processed/model_input.csv`(`config.yaml`의 `model_input`)를 만든다.
- **목표가 표 (03)**: `data/processed/articles_clean.csv`(정리된 기사) → `tables/tp_monthly.csv`. 기사가 없는 환경에서는 건너뛰고 저장소의 표를 쓴다. 2022–2025 기사가 생기면 `config.yaml`의 `articles`에 추가해 다시 돌리고, 바뀐 표를 커밋한다.

처음 한 번은 Docker 밖에서 `python src/make_trading_days.py`(pykrx 필요)로 `tables/last_trading_days.csv`를 만든다. 파일별 설명은 `Quant/tables/README.md`에 있다.

## 규칙

- `data/`는 Git에 올리지 않는다. 배포 데이터는 드라이브 공개 폴더, 빅카인즈 원문은 비공개 폴더에 둔다.
- test(2022~2025)는 학습·전처리·변수 선택에 쓰지 않는다. 기간은 `config.yaml`에서만 정한다.
- `main`에 직접 올리지 않는다. 브랜치 → Pull Request → 1명 승인 후 병합.
- 노트북은 커밋 전에 Clear All Outputs.

## 최종 예측 (prediction.csv)

- `prediction.csv`는 `python run_all.py --final`일 때만 생긴다. "test(2022~2025)는 한 번만" 규칙 때문에 일부러 이렇게 했다. 기본 실행은 검증 IC 표와 그림까지만 만든다.
- 2021 가짜 test 리허설은 `config.yaml`의 test를 `["2021-01", "2021-12"]`로 바꾸고 `--final`을 붙이면 된다. 코드는 고칠 필요가 없다. 끝나면 test를 원래대로 되돌린다.
