"""[01] 배포 데이터 불러오기·결합·시점 정렬   담당: A

입력  data/raw/  (warmup·train CSV, 행=월 열=6자리 종목코드)
출력  data/processed/panel.parquet  (long 형식: month, code, ret, me, ...)
형식  month = "YYYY-MM" 문자열, code = 6자리 문자열, 값 컬럼은 float64
"""

from pathlib import Path
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gdown
import yaml

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
RAW = Path(cfg["data_dir"]) / "raw"
PROC = Path(cfg["data_dir"]) / "processed"


# --------------------------------------------------
# 1. 데이터 로드
# --------------------------------------------------

def load_provided_data(folder_url: str,) -> dict[str, pd.DataFrame]:
    
    RAW.mkdir(parents=True, exist_ok=True)

    gdown.download_folder(
        url=folder_url,
        output=str(RAW),
        quiet=False
    )

    return {
        file.stem: pd.read_csv(file, encoding="utf-8-sig")
        for file in RAW.glob("*.csv")
        if "warmup" in file.stem.lower() or "train" in file.stem.lower()
    }


def load_provided_data_from_local() -> dict[str, pd.DataFrame]:
    
    return {
        file.stem: pd.read_csv(file, encoding="utf-8-sig")
        for file in RAW.glob("*.csv")
        if "warmup" in file.stem.lower() or "train" in file.stem.lower()
    }


def load_long_data_from_local() -> pd.DataFrame:
    """Load the long-format data from local storage."""
    long_file = PROC / "warmup_train_long.csv"
    if not long_file.exists():
        raise FileNotFoundError(f"{long_file} does not exist. Please run the data processing pipeline first.")
    
    return pd.read_csv(long_file, encoding="utf-8-sig", parse_dates=["Month"])


# ==================================================
# 2. 데이터 구조 검증 및 통일
# ==================================================

def check_data_structure(data_dict):

    reference_name = list(data_dict.keys())[0]
    reference_df = data_dict[reference_name]

    all_match = True

    for name, df in data_dict.items():

        # 크기 확인
        same_shape = df.shape == reference_df.shape

        # 컬럼명 및 순서 확인
        same_columns = df.columns.equals(reference_df.columns)

        # 날짜 확인 (첫 번째 컬럼 기준)
        same_dates = df.iloc[:, 0].equals(
            reference_df.iloc[:, 0]
        )

        # 결과 출력
        print(f"\n[{name}]")
        print(f"Shape 일치: {same_shape}")
        print(f"Columns 일치: {same_columns}")
        print(f"Dates 일치: {same_dates}")

        if not (same_shape and same_columns and same_dates):
            all_match = False

    print("\n" + "=" * 35)

    if all_match:
        print("모든 DataFrame의 구조가 동일합니다.")
    else:
        print("DataFrame 간 구조 차이가 발견되었습니다.")

    return all_match


def rename_dict_keys(
    df_dict: dict[str, pd.DataFrame],
    data_type: Literal["warmup", "train"],
) -> dict[str, pd.DataFrame]:
   
    rename_maps = {
        "warmup": {
            "warmup_2000__adjusted_monthly_return": "adjusted_monthly_return",
            "warmup_2000__company_name": "company_name",
            "warmup_2000__isin": "isin",
            "warmup_2000__raw_month_end_market_cap": "raw_month_end_market_cap",
            "warmup_2000__raw_month_end_price": "raw_month_end_price",
            "warmup_2000__raw_month_end_shares_outstanding": "raw_month_end_shares_outstanding",
        },
        "train": {
            "train_2001_2021__adjusted_monthly_return": "adjusted_monthly_return",
            "train_2001_2021__company_name": "company_name",
            "train_2001_2021__isin": "isin",
            "train_2001_2021__raw_month_end_market_cap": "raw_month_end_market_cap",
            "train_2001_2021__raw_month_end_price": "raw_month_end_price",
            "train_2001_2021__raw_month_end_shares_outstanding": "raw_month_end_shares_outstanding",
        },
    }

    if data_type not in rename_maps:
        raise ValueError(f"지원하지 않는 데이터 종류입니다: {data_type}")

    rename_map = rename_maps[data_type]

    return {
        rename_map.get(key, key): value
        for key, value in df_dict.items()
    }


# ==================================================
# 3. Long-format 변환
# ==================================================

def convert_to_long(
    df_dict: dict[str, pd.DataFrame],
) -> pd.DataFrame:

    long_dfs = []

    for variable, df in df_dict.items():

        # Wide → Long
        long_df = df.melt(
            id_vars=df.columns[0],
            var_name="Stock_Code",
            value_name=variable
        )

        # 날짜 컬럼명 통일
        long_df = long_df.rename(
            columns={df.columns[0]: "Month"}
        )

        long_dfs.append(long_df)

    # 첫 번째 DataFrame을 기준으로 병합
    result = long_dfs[0]

    for long_df in long_dfs[1:]:
        result = result.merge(
            long_df,
            on=["Month", "Stock_Code"],
            how="outer",
            validate="one_to_one"
        )

    # 날짜 형식 변환
    result["Month"] = pd.to_datetime(result["Month"])

    # 정렬
    result = result.sort_values(
        ["Month", "Stock_Code"]
    ).reset_index(drop=True)
    
    return result


def combine_and_save_long_data(
    warmup_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine warmup and train data, then save the result as CSV."""
    if not warmup_df.columns.equals(train_df.columns):
        raise ValueError("warmup and train DataFrames must have identical columns.")

    combined_df = pd.concat(
        [warmup_df, train_df],
        ignore_index=True,
    ).sort_values(
        ["Month", "Stock_Code"]
    ).reset_index(drop=True)

    expected_rows = len(warmup_df) + len(train_df)
    actual_rows = len(combined_df)
    if actual_rows != expected_rows:
        raise ValueError(
            f"행 개수 불일치: 예상 {expected_rows}행, 실제 {actual_rows}행"
        )
    print(f"행 개수 검증 통과: {actual_rows}행 = {len(warmup_df)}행 + {len(train_df)}행")

    PROC.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(PROC / "warmup_train_long.csv", index=False)
    return combined_df


# ==================================================
# 4. 배포 데이터 현황 진단
# ==================================================

def check_duplicate_records(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Month와 Stock_Code 조합이 중복된 레코드를 확인한다.

    Output:
        중복 키에 해당하는 모든 행
    """
    key_columns = ["Month", "Stock_Code"]
    missing_columns = [column for column in key_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {', '.join(missing_columns)}")

    duplicates = df.loc[
        df.duplicated(subset=key_columns, keep=False)
    ].sort_values(key_columns).reset_index(drop=True)

    if duplicates.empty:
        print("중복 Month + Stock_Code 레코드가 없습니다.")
    else:
        duplicate_key_count = duplicates[key_columns].drop_duplicates().shape[0]
        print(
            f"중복 Month + Stock_Code 조합 {duplicate_key_count}개, "
            f"관련 행 {len(duplicates)}개를 발견했습니다."
        )

    return duplicates


def check_missing_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    식별 컬럼을 제외한 각 컬럼의 결측치 개수와 비율을 출력한다.

    Output:
        컬럼별 결측치 개수와 비율이 담긴 DataFrame
    """
    excluded_columns = {"month", "stock_code", "company_name", "isin"}
    columns_to_check = [
        column for column in df.columns
        if column.lower() not in excluded_columns
    ]

    missing_count = df[columns_to_check].isna().sum()
    missing_summary = pd.DataFrame({
        "Missing_Count": missing_count,
        "Missing_Rate(%)": missing_count.div(len(df)).mul(100),
    })

    print(missing_summary.to_string())
    
    all_missing_rows = (
        df[columns_to_check].isna().all(axis=1).sum()
        if columns_to_check
        else 0
    )
    print(f"검사 대상 컬럼이 모두 결측인 행 수: {all_missing_rows}\n")

    return missing_summary


def check_outliers(
    df: pd.DataFrame,
) -> pd.DataFrame:

    # 1. 데이터 정렬
    df = df.sort_values(
        ["Stock_Code", "Month"]
    ).copy()


    # 2. 논리적 범위 검사
    df["price_invalid"] = (
        df["raw_month_end_price"] <= 0
    )

    df["shares_invalid"] = (
        df["raw_month_end_shares_outstanding"] <= 0
    )

    df["market_cap_invalid"] = (
        df["raw_month_end_market_cap"] <= 0
    )

    df["return_invalid"] = (
        df["adjusted_monthly_return"] < -1
    )


    # 3. 종목별 IQR 이상치 탐지
    def group_iqr_outlier(series, min_obs=12):

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if series.notna().sum() < min_obs:
            return pd.Series(False, index=series.index)

        if pd.isna(iqr) or iqr == 0:
            return pd.Series(False, index=series.index)

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        return (
            (series < lower) |
            (series > upper)
        ).fillna(False)
    
    df["return_iqr_outlier"] = (
        df.groupby("Stock_Code")[
            "adjusted_monthly_return"
        ].transform(group_iqr_outlier)
    )


    # 4. 시계열 수준 이상치 탐지
    level_cols = {
        "price": "raw_month_end_price",
        "shares": "raw_month_end_shares_outstanding",
        "market_cap": "raw_month_end_market_cap",
    }

    def rolling_iqr_outlier(
        series,
        window=12,
        min_periods=6
    ):
        log_series = np.log(
            series.where(series > 0)
        )

        history = log_series.shift(1)

        q1 = history.rolling(
            window=window,
            min_periods=min_periods
        ).quantile(0.25)

        q3 = history.rolling(
            window=window,
            min_periods=min_periods
        ).quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        flag = (
            ((log_series < lower) |
             (log_series > upper)) &
            (iqr > 0)
        )

        return flag.fillna(False)
    
    for name, col in level_cols.items():

        df[f"{name}_level_outlier"] = (
            df.groupby("Stock_Code")[col]
            .transform(rolling_iqr_outlier)
        )


    # 5. 전월 대비 변화율 계산
    for name, col in level_cols.items():

        change_col = f"{name}_change_rate"

        df[change_col] = (
            df.groupby("Stock_Code")[col]
            .pct_change(fill_method=None)
        )

        df[f"{name}_change_outlier"] = (
            df.groupby("Stock_Code")[
                change_col
            ].transform(group_iqr_outlier)
        )


    # 6. 시가총액 정합성 검사

    calculated_market_cap = (
        df["raw_month_end_price"] *
        df["raw_month_end_shares_outstanding"]
    )

    df["market_cap_consistency_ratio"] = (
        df["raw_month_end_market_cap"] /
        calculated_market_cap.where(
            calculated_market_cap > 0
        )
    )

    df["market_cap_consistency_outlier"] = (
        (
            df["market_cap_consistency_ratio"] - 1
        ).abs() > 0.01
    ).fillna(False)


    # 7. 종합 이상치 여부
    flag_cols = [
        "price_invalid",
        "shares_invalid",
        "market_cap_invalid",
        "return_invalid",
        "return_iqr_outlier",
        "price_level_outlier",
        "shares_level_outlier",
        "market_cap_level_outlier",
        "price_change_outlier",
        "shares_change_outlier",
        "market_cap_change_outlier",
        "market_cap_consistency_outlier",
    ]

    df["outlier_any"] = df[flag_cols].any(axis=1)


    # 8. 이상치 탐지 사유
    reason_labels = {
        "price_invalid": "종가 논리적 오류",
        "shares_invalid": "유통 주식 수 논리적 오류",
        "market_cap_invalid": "시가총액 논리적 오류",
        "return_invalid": "수익률 논리적 오류",
        "return_iqr_outlier": "종목별 수익률 IQR 이상치",
        "price_level_outlier": "종가 시계열 수준 이상치",
        "shares_level_outlier": "주식 수 시계열 수준 이상치",
        "market_cap_level_outlier": "시가총액 시계열 수준 이상치",
        "price_change_outlier": "종가 변화율 이상치",
        "shares_change_outlier": "주식 수 변화율 이상치",
        "market_cap_change_outlier": "시가총액 변화율 이상치",
        "market_cap_consistency_outlier": "시가총액 정합성 이상치",
    }

    df["outlier_reason"] = ""

    for col, label in reason_labels.items():

        mask = df[col]

        df.loc[mask, "outlier_reason"] = (
            df.loc[mask, "outlier_reason"]
            .apply(
                lambda x: f"{x}, {label}" if x else label
            )
        )

    return df


def show_outlier_summary(outlier_df: pd.DataFrame) -> None:

    # 이상치가 있는 행만 추출
    outlier_rows = outlier_df[
    outlier_df["outlier_any"]
    ].copy()


    # 이상치 유형별 개수
    outlier_summary = (
        outlier_df[
            [col for col in outlier_df.columns
            if col.endswith("_outlier")]
        ]
        .sum()
        .sort_values(ascending=False)
    )
    print(outlier_summary)

    outlier_summary = (
            outlier_df[
                [col for col in outlier_df.columns
                if col.endswith("_invalid")]
            ]
            .sum()
            .sort_values(ascending=False)
        )
    print(outlier_summary)


    # 이상치 사례 확인
    print(
        outlier_rows[
            [
                "Month",
                "Stock_Code",
                "company_name",
                "adjusted_monthly_return",
                "outlier_reason",
            ]
        ].head(7)
    )


def show_monthly_return_trend(warmup_train_long: pd.DataFrame) -> None:
    """Plot the monthly median adjusted return and its interquartile range."""
    
    required_columns = {"Month", "adjusted_monthly_return"}
    missing_columns = required_columns.difference(warmup_train_long.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    returns = warmup_train_long.loc[
        warmup_train_long["adjusted_monthly_return"] >= -1,
        ["Month", "adjusted_monthly_return"],
    ]
    monthly_summary = (
        returns.groupby("Month")["adjusted_monthly_return"]
        .agg(
            median="median",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
        )
        .dropna()
        .sort_index()
    )
    if monthly_summary.empty:
        raise ValueError("No valid monthly returns are available to plot.")

    months = monthly_summary.index
    plt.figure(figsize=(12, 6))
    plt.plot(
        months,
        monthly_summary["median"] * 100,
        label="Median",
        color="navy",
    )
    plt.fill_between(
        months,
        monthly_summary["q1"] * 100,
        monthly_summary["q3"] * 100,
        alpha=0.2,
        color="steelblue",
        label="25th-75th percentile",
    )
    plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.title("Monthly Adjusted Return Trend")
    plt.xlabel("Month")
    plt.ylabel("Adjusted return (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==================================================
# 5. 정제
# ==================================================

def handle_duplicate_records(
    df: pd.DataFrame,
    variable_name: str,
) -> pd.DataFrame:
    """중복 관측치를 처리한다."""
    pass


def handle_missing_values(
    df: pd.DataFrame,
    variable_name: str,
) -> pd.DataFrame:
    """결측치는 채우지 않고 NaN으로 둔다.

    빈칸 = 그 달 미상장·거래 없음(배포 README). 이전 값으로 채우면 수익률·모멘텀이 왜곡된다.
    대체(월별 중앙값)와 결측 표시 열은 03 rank_transform에서 한 번만 만든다.
    """
    pass


def handle_outliers(
    df: pd.DataFrame,
    variable_name: str,
) -> pd.DataFrame:
    """원자료 값은 바꾸지 않는다(이상치 후보는 check_outliers로 기록만).

    극단값 처리는 02에서 계산한 변수에 월별 단면 기준으로만 한다(기간을 합친 경계값은 test 누수).
    """
    pass


# ==================================================
# 6. 전체 파이프라인
# ==================================================

def long_data_pipeline() -> pd.DataFrame:
    """Load and combine warmup and train data into long format."""

    # folder_url = "https://drive.google.com/drive/folders/1qWVfJ50mYzVxwB8wlYn4MKfBfPC6speu?usp=sharing"
    # provided_data = load_provided_data(folder_url)
    provided_data = load_provided_data_from_local()

    warmup_dict = {
        key: value
        for key, value in provided_data.items()
        if 'warmup' in key
    }
    
    train_dict = {
        key: value
        for key, value in provided_data.items()
        if 'train' in key
    }

    check_data_structure(warmup_dict)
    check_data_structure(train_dict)

    warmup_dict = rename_dict_keys(warmup_dict, "warmup")
    train_dict = rename_dict_keys(train_dict, "train")

    warmup_long = convert_to_long(warmup_dict)
    train_long = convert_to_long(train_dict)
    warmup_train_long = combine_and_save_long_data(warmup_long, train_long)

    duplicate_records = check_duplicate_records(warmup_train_long)
    if not duplicate_records.empty:
        print(duplicate_records)

    return warmup_train_long


def eda_pipeline(warmup_train_long: pd.DataFrame) -> None:
    """Perform exploratory data analysis on the long-format data."""
    check_missing_values(warmup_train_long)
    outlier_df = check_outliers(warmup_train_long)
    show_outlier_summary(outlier_df)
    show_monthly_return_trend(warmup_train_long)


def main():

    # warmup_train_long = long_data_pipeline()
    warmup_train_long = load_long_data_from_local()

    eda_pipeline(warmup_train_long)

    
if __name__ == "__main__":
    main()
