# FDI Team04: KOSPI 다음 달 수익률 순위 예측

금융데이터인텔리전스(2026-2) 팀 프로젝트. 매월 말까지 알 수 있는 정보로 다음 달 종목별 수익률 순위를 예측한다.

## 실행 (Docker)

Docker Desktop을 켠 상태에서, 모든 명령은 저장소 맨 위 폴더(`docker-compose.yml`이 있는 곳)에서 입력한다.

### 빌드

```bash
docker compose up --build
```

처음에는 Python과 torch를 받느라 몇 분 걸린다. 마지막에 `ERROR` 없이 끝나면 성공이다.

### 확인

```bash
docker compose up --build
```

### 나가기
```bash
ctrl + C
```

`Python 3.11.x`가 나오고, `▶ src/01_load_clean.py`부터 `▶ src/05_evaluate.py`까지 출력되면 준비 끝이다.

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
- FinBERT 모델은 처음 실행할 때 한 번만 받고 저장해 둔다.
- Docker는 **CPU만** 쓴다. FinBERT 점수 계산은 시간이 걸리지만 결과를 캐시 파일로 저장하므로 한 번만 돌리면 된다.
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
    └── data/        (Git 제외)
```

| 경로 | 내용 | 담당 |
|---|---|---|
| `docker-compose.yml` | Docker 실행 설정: 폴더 연결, `.env`, 실행 명령 | 공통 |
| `Quant/Dockerfile` | Docker 이미지: Python 3.11 + 패키지 설치 | 공통 |
| `Quant/requirements.txt` | 패키지 버전 고정 | 공통 |
| `Quant/config.yaml` | 기간 구분, 데이터 경로, 시드, 텍스트 모델 | 공통 |
| `Quant/run_all.py` | `src/01` → `05` 순서 실행 | 공통 |
| `Quant/src/01_load_clean.py` | 배포 데이터 결합·시점 정렬 | A |
| `Quant/src/02_features.py` | 정형 변수 (H1~H7) | A |
| `Quant/src/03_text_features.py` | 텍스트 감성 점수·결합·순위 변환 (H8) | B-2 |
| `Quant/src/04_model.py` | 모형 학습 | 미정 |
| `Quant/src/05_evaluate.py` | 월별 Spearman IC | 미정 |
| `Quant/data/` | `raw/` 원자료, `processed/` 결과 (Git 제외, 각자 드라이브에서 받음) | 각자 |
| `docs/` | 정의서, 자료 메모, 재수집 방법 | 각자 |

## 규칙

- `data/`는 Git에 올리지 않는다. 배포 데이터는 드라이브 공개 폴더, 빅카인즈 원문은 비공개 폴더에 둔다.
- test(2022~2025)는 학습·전처리·변수 선택에 쓰지 않는다. 기간은 `config.yaml`에서만 정한다.
- `main`에 직접 올리지 않는다. 브랜치 → Pull Request → 1명 승인 후 병합.
- 노트북은 커밋 전에 Clear All Outputs.
