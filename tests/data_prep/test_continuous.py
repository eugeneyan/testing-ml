import pandas as pd
import pytest

from src.data_prep.continuous import fill_numeric


@pytest.fixture
def dummy_df():
    int_col = [1, None, 4, 2, 100]
    float_col = [100.234, 0.0, None, None, 9.19]
    df_dict = {'int': int_col, 'float': float_col}
    df = pd.DataFrame(df_dict)
    return df


def test_fill_numeric_default(dummy_df):
    result_int = fill_numeric(dummy_df, col='int')
    result_float = fill_numeric(dummy_df, col='float')
    assert result_int['int'].tolist() == [1.0, 3.0, 4.0, 2.0, 100.0]
    assert result_float['float'].tolist() == [100.234, 0.0, 9.19, 9.19, 9.19]


def test_fill_numeric_median(dummy_df):
    result_int = fill_numeric(dummy_df, col='int', fill_type='median')
    result_float = fill_numeric(dummy_df, col='float', fill_type='median')
    assert result_int['int'].tolist() == [1.0, 3.0, 4.0, 2.0, 100.0]
    assert result_float['float'].tolist() == [100.234, 0.0, 9.19, 9.19, 9.19]


def test_fill_numeric_mean(dummy_df):
    result_int = fill_numeric(dummy_df, col='int', fill_type='mean')
    result_float = fill_numeric(dummy_df, col='float', fill_type='mean')
    assert result_int['int'].tolist() == [1.0, 26.75, 4.0, 2.0, 100.0]
    assert result_float['float'].tolist() == [100.234, 0.0, 36.474666666666664, 36.474666666666664, 9.19]


def test_fill_numeric_minus1(dummy_df):
    result_int = fill_numeric(dummy_df, col='int', fill_type='-1')
    result_float = fill_numeric(dummy_df, col='float', fill_type='-1')
    assert result_int['int'].tolist() == [1.0, -1, 4.0, 2.0, 100.0]
    assert result_float['float'].tolist() == [100.234, 0.0, -1.0, -1.0, 9.19]


def test_fill_numeric_not_implemented(dummy_df):
    with pytest.raises(NotImplementedError):
        fill_numeric(dummy_df, col='int', fill_type='random')
