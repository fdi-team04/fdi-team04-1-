"""[06] 방향 확인과 해석   입력: model_input.parquet   출력: data/processed/*.csv, *.png"""

def fama_macbeth(df, cols):
    """04에서 옮겨 옴. 월별 OLS → 계수 평균, t = 평균 / 표준오차 (|t| ≥ 2)"""

def quintile_table(df, col):
    """col 상·하위 20%의 다음 달 수익률, 차이, t값 (retail_nb, foreign_nb, inst_nb)"""

def h4_table(df):
    """topic_tone 상·하 절반 × retail_nb 상·하 절반 → 2×2 다음 달 수익률 (no_news = 0인 행만)"""

def shap_dependence(model, X):
    """topic_tone의 SHAP 값을 retail_nb로 색칠한 그림 (H4 모양 확인)"""