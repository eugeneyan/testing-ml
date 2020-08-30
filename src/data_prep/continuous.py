"""
Data preparation methods for continuous variables.
"""
import pandas as pd


def fill_numeric(df: pd.DataFrame, col: str, fill_type: str = 'median') -> pd.DataFrame:
    """Fills missing values in numeric column specified.

    Args:
        df: DataFrame to fill
        col: Column in DataFrame to fill
        fill_type: How to fill the data. Supported types: "mean", "median", "-1"

    Returns:
        A DataFrame with numeric_col filled.
    """
    if fill_type == 'median':
        fill_value = df[col].median()  # type: float
    elif fill_type == 'mean':
        fill_value = df[col].mean()
    elif fill_type == '-1':
        fill_value = -1
    else:
        raise NotImplementedError('Valid fill_type options are "mean", "median", "-1')

    df.loc[df[col].isnull(), col] = fill_value
    return df
