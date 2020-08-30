import pandas as pd
import pytest

from src.data_prep.categorical import lowercase_string, lowercase_column, extract_title


@pytest.fixture
def dummy_df():
    string_col = ['Futrelle, Mme. Jacques Heath (Lily May Peel)',
                  'Johnston, Ms. Catherine Helen "Carrie"',
                  'Sloper, Mr. William Thompson',
                  'Ostby, Lady. Engelhart Cornelius',
                  'Backstrom, Major. Karl Alfred (Maria Mathilda Gustafsson)']
    df_dict = {'string': string_col}
    df = pd.DataFrame(df_dict)
    return df


@pytest.fixture
def lowercased_df():
    string_col = ['futrelle, mme. jacques heath (lily may peel)',
                  'johnston, ms. catherine helen "carrie"',
                  'sloper, mr. william thompson',
                  'ostby, lady. engelhart cornelius',
                  'backstrom, major. karl alfred (maria mathilda gustafsson)']
    df_dict = {'string': string_col}
    df = pd.DataFrame(df_dict)
    return df


def test_lowercase_string():
    assert lowercase_string('abc') == 'abc'
    assert lowercase_string('ABC') == 'abc'
    assert lowercase_string('Abc123') == 'abc123'
    assert lowercase_string(1) is None


def test_lowercase_column(dummy_df, lowercased_df):
    actual_df = lowercase_column(dummy_df, col='string')
    assert actual_df.equals(lowercased_df)


def test_extract_title(lowercased_df):
    result = extract_title(lowercased_df, col='string')
    assert result['title'].tolist() == ['mme', 'ms', 'mr', 'lady', 'major']

    title_replacement = {'mme': 'mrs', 'ms': 'miss', 'lady': 'rare', 'major': 'rare'}
    result = extract_title(lowercased_df, col='string', replace_dict=title_replacement)
    assert result['title'].tolist() == ['mrs', 'miss', 'mr', 'rare', 'rare']
