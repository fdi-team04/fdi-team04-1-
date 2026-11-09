"""[02] 정형 변수 계산 (H1~H7)   담당: A

입력  data/processed/panel.parquet
출력  data/processed/features.parquet  (month, code, 변수들, target = 다음 달 수익률)
"""

from pathlib import Path
from dotenv import load_dotenv
import hashlib
import os

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

krx_id = os.getenv("KRX_ID")
krx_pw = os.getenv("KRX_PW")

if not krx_id or not krx_pw:
    raise ValueError(
        "KRX_ID 또는 KRX_PW 환경 변수가 설정되지 않았습니다."
    )

import pandas as pd
import gdown
import time
import re
from tqdm.auto import tqdm
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
with CONFIG_PATH.open(encoding="utf-8") as config_file:
    cfg = yaml.safe_load(config_file)
RAW = Path(cfg["data_dir"]) / "raw"
PROC = Path(cfg["data_dir"]) / "processed"


# ============================================================
# 1. Load data
# ============================================================

def load_long_data_from_local() -> pd.DataFrame:

    long_file = PROC / "panel.parquet"
    if not long_file.exists():
        raise FileNotFoundError(f"{long_file} does not exist. Please run the data processing pipeline first.")
    
    return pd.read_parquet(long_file)


def load_risk_free_rate(data_dir: Path) -> pd.DataFrame:
    """Load and monthly risk-free rate data"""
    pass


def get_pbr(data_dir: Path) -> pd.DataFrame:
    """Get monthly PBR data"""
    pass


def make_ticker_df(long_df: pd.DataFrame, ticker_col: str = "Stock_Code") -> pd.DataFrame:
    """Return one row per non-null ticker code from the long-format data."""
    if not isinstance(long_df, pd.DataFrame):
        raise TypeError("long_df는 pandas DataFrame이어야 합니다.")

    if ticker_col not in long_df.columns:
        raise ValueError(f"'{ticker_col}' 컬럼이 존재하지 않습니다.")

    return (
        long_df.loc[long_df[ticker_col].notna(), [ticker_col]]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def get_monthly_investor_data(
    ticker_df,
    start_date="20000101",
    end_date="20211231",
    ticker_col="Stock_Code",
    sleep_sec=1.0,
    progress_every=50
):
    """
    기존 종목 데이터프레임을 입력받아 KOSPI 개별 종목의
    개인 및 외국인 월별 순매수 거래대금을 수집한다.

    Parameters
    ----------
    ticker_df : pandas.DataFrame
        종목 목록이 들어 있는 데이터프레임.
    start_date : str
        조회 시작일 (YYYYMMDD).
    end_date : str
        조회 종료일 (YYYYMMDD).
    ticker_col : str
        종목코드가 들어 있는 컬럼명.
    sleep_sec : float
        종목별 조회 사이 대기 시간(초).
    progress_every : int
        진행 상황을 출력할 간격.

    Returns
    -------
    monthly_df : pandas.DataFrame
        month, ticker, 개인, 외국인 컬럼을 가진 월별 데이터.
    failed_df : pandas.DataFrame
        조회 실패 또는 빈 결과가 발생한 종목과 오류 내용.
    """

    # --------------------------------------------------
    # 1. 입력값 검사 및 종목코드 목록 생성
    # --------------------------------------------------

    if not isinstance(ticker_df, pd.DataFrame):
        raise TypeError("ticker_df는 pandas DataFrame이어야 합니다.")

    if ticker_col not in ticker_df.columns:
        raise ValueError(f"'{ticker_col}' 컬럼이 존재하지 않습니다.")

    start = pd.to_datetime(start_date, format="%Y%m%d")
    end = pd.to_datetime(end_date, format="%Y%m%d")

    if start > end:
        raise ValueError("start_date는 end_date보다 늦을 수 없습니다.")

    def normalize_ticker(value):
        """종목코드를 6자리 문자열로 변환한다."""

        if pd.isna(value):
            return None

        ticker = str(value).strip()

        # 일부 데이터에서 나타나는 A005930 형식 처리
        if ticker.startswith("A") and ticker[1:].isdigit():
            ticker = ticker[1:]

        # 숫자형으로 저장되어 5930.0이 된 경우 처리
        if re.fullmatch(r"\d+\.0", ticker):
            ticker = ticker[:-2]

        if ticker.isdigit():
            return ticker.zfill(6)

        return ticker if ticker else None

    tickers = (
        ticker_df[ticker_col]
        .apply(normalize_ticker)
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    if not tickers:
        raise ValueError("유효한 종목코드가 없습니다.")

    print(f"총 조회 대상 종목 수: {len(tickers):,}")
    print(f"조회 기간: {start_date} ~ {end_date}")

    # --------------------------------------------------
    # 2. 종목별 일별 데이터 조회 및 월별 집계
    # --------------------------------------------------

    monthly_results = []
    failed_results = []

    try:
        from pykrx import stock
    except Exception as e:
        failed_results.extend(
            {"ticker": ticker, "error": f"pykrx 초기화 실패: {e!r}"}
            for ticker in tickers
        )
        monthly_df = pd.DataFrame(
            columns=["month", "ticker", "개인", "외국인"]
        )
        failed_df = pd.DataFrame(
            failed_results,
            columns=["ticker", "error"]
        )
        print("\n수집 완료")
        print(f"성공 종목 수: 0")
        print(f"실패 종목 수: {len(failed_df):,}")
        print(f"최종 데이터 행 수: 0")
        return monthly_df, failed_df

    for i, ticker in enumerate(tickers, start=1):

        try:
            # 종목당 전체 기간을 한 번에 조회
            daily_df = stock.get_market_trading_value_by_date(
                fromdate=start_date,
                todate=end_date,
                ticker=ticker,
                on="순매수",
                detail=True,
                freq="d"
            )

            # 데이터가 없는 경우
            if daily_df is None or daily_df.empty:
                failed_results.append({
                    "ticker": ticker,
                    "error": "조회 결과 없음"
                })
                continue

            # 개인과 외국인 컬럼 확인
            required_cols = ["개인", "외국인"]

            if not all(col in daily_df.columns for col in required_cols):
                failed_results.append({
                    "ticker": ticker,
                    "error": f"필요한 컬럼 누락: {daily_df.columns.tolist()}"
                })
                continue

            # 개인과 외국인 데이터만 사용
            daily_df = daily_df[required_cols].copy()

            # 날짜 인덱스 변환
            daily_df.index = pd.to_datetime(daily_df.index)

            # 숫자형 변환
            daily_df = daily_df.apply(
                pd.to_numeric,
                errors="coerce"
            )

            # ------------------------------------------
            # 3. 매월 달력상 마지막 날짜 제외
            # ------------------------------------------

            daily_df = daily_df[
                daily_df.index.day < daily_df.index.days_in_month
            ]

            if daily_df.empty:
                failed_results.append({
                    "ticker": ticker,
                    "error": "월말 제외 후 데이터 없음"
                })
                continue

            # ------------------------------------------
            # 4. 월별 합계 계산
            # ------------------------------------------

            monthly = (
                daily_df
                .groupby(daily_df.index.to_period("M"))
                .sum(min_count=1)
            )

            monthly.index.name = "month"
            monthly = monthly.reset_index()

            # YYYY-MM 문자열로 변환
            monthly["month"] = monthly["month"].astype(str)

            # 종목코드 추가
            monthly["ticker"] = ticker

            # 컬럼 순서 정리
            monthly = monthly[
                ["month", "ticker", "개인", "외국인"]
            ]

            monthly_results.append(monthly)

        except Exception as e:
            failed_results.append({
                "ticker": ticker,
                "error": repr(e)
            })

        finally:
            # 요청 간 대기
            if sleep_sec > 0:
                time.sleep(sleep_sec)

        # 진행 상황 출력
        if i % progress_every == 0 or i == len(tickers):
            print(
                f"진행: {i:,}/{len(tickers):,} "
                f"| 성공: {len(monthly_results):,} "
                f"| 실패: {len(failed_results):,}"
            )

    # --------------------------------------------------
    # 5. 최종 데이터프레임 생성
    # --------------------------------------------------

    if monthly_results:
        monthly_df = pd.concat(
            monthly_results,
            ignore_index=True
        )

        monthly_df = (
            monthly_df
            .sort_values(["month", "ticker"])
            .reset_index(drop=True)
        )

    else:
        monthly_df = pd.DataFrame(
            columns=["month", "ticker", "개인", "외국인"]
        )

    failed_df = pd.DataFrame(
        failed_results,
        columns=["ticker", "error"]
    )

    print("\n수집 완료")
    print(f"성공 종목 수: {len(monthly_results):,}")
    print(f"실패 종목 수: {len(failed_df):,}")
    print(f"최종 데이터 행 수: {len(monthly_df):,}")

    return monthly_df, failed_df


# ============================================================
# 2. Standardize and join external data
# ============================================================

def standardize_join_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize stock codes and month columns for joining."""
    pass


def validate_join_keys(
    df: pd.DataFrame,
    key_cols: list[str],
) -> pd.DataFrame:
    """Check missing, invalid, and duplicate join keys."""
    pass


def join_external_data(
    distributed_df: pd.DataFrame,
    risk_free_df: pd.DataFrame,
    pbr_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join external data using the appropriate stock-code and month keys."""
    pass


def check_join_quality(
    df: pd.DataFrame,
    expected_rows: int,
) -> pd.DataFrame:
    """Check row-count changes, unmatched records, duplicates, and missingness after joins."""
    pass


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Leave missing external data as NaN (no filling here).

    Median fill and *_missing flags are created once in 03 rank_transform.
    """
    pass


# ============================================================
# 3. Generate derived features
# ============================================================

def calculate_market_beta(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate market beta for each stock and month."""
    pass


def calculate_book_to_market(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the B/M ratio from the available PBR data."""
    pass


def calculate_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate stock momentum using historical returns over the chosen lookback window."""
    pass


def calculate_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate stock-return volatility for each stock and month."""
    pass


def calculate_stock_issuance(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate stock issuance for each stock and month."""
    pass


def calculate_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create target = next-month adjusted return r[i,t+1], by code.

    The column name must be exactly `target`: 03_text_features drops only that name
    from the model inputs, so any other name would leak next-month returns into the features.
    """
    pass


# ============================================================
# 4. Inspect and clean derived features
# ============================================================

def generate_feature_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize feature statistics, missingness, duplicates, and potential outliers."""
    pass


def handle_feature_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Leave missing derived features as NaN (no filling here).

    Median fill and *_missing flags are created once in 03 rank_transform;
    filling earlier would make every *_missing flag 0.
    """
    pass


def handle_feature_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature-specific outlier treatment while preserving valid observations.

    Use monthly cross-sectional cutoffs only (e.g. 1%/99% within each month);
    cutoffs pooled across months would leak test-period information.
    """
    pass


def validate_final_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, key uniqueness, data types, and final feature quality."""
    pass


# ============================================================
# 6. Pipeline
# ============================================================

def main():
    
    warmup_train_long = load_long_data_from_local()
    ticker_df = make_ticker_df(warmup_train_long, ticker_col="Stock_Code")
    
    investor_df, failed_investor_df = get_monthly_investor_data(
        ticker_df=ticker_df,
        start_date="20000101",
        end_date="20211231",
        ticker_col="Stock_Code",
        sleep_sec=1.0,
        progress_every=50
    )
    
    PROC.mkdir(parents=True, exist_ok=True)
    investor_df.to_csv(PROC / "investor.csv", index=False, encoding="utf-8-sig")
    failed_investor_df.to_csv(PROC / "failed_investor.csv", index=False, encoding="utf-8-sig")

    print(investor_df.head())

if __name__ == "__main__":
    main()
