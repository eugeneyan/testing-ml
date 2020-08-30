"""
Data preparation for Titanic data. Includes train-test split and creating of features and labels.
"""
from typing import Tuple

import pandas as pd
from numpy import array
from sklearn.model_selection import train_test_split

from src.data_prep.categorical import lowercase_column, extract_title
from src.data_prep.continuous import fill_numeric

title_mapping = {'mme': 'mrs',
                 'ms': 'miss',
                 'mlle': 'miss',
                 'lady': 'rare',
                 'countess': 'rare',
                 'capt': 'rare',
                 'col': 'rare',
                 'don': 'rare',
                 'dr': 'rare',
                 'major': 'rare',
                 'rev': 'rare',
                 'sir': 'rare',
                 'jonkheer': 'rare',
                 'dona': 'rare'}
title_to_int = {'mr': 1, 'mrs': 2, 'miss': 3, 'master': 4, 'rare': 5, 'NA': -1}
sex_to_int = {'female': 1, 'male': 0, 'NA': -1}
port_to_int = {'s': 0, 'c': 1, 'q': 2, 'NA': -1}


def load_df() -> pd.DataFrame:
    """Return raw Titanic Data

    Returns:
        Titanic data loaded from data/titanic.csv
    """
    return pd.read_csv('data/titanic.csv')


def prep_df(df: pd.DataFrame) -> pd.DataFrame:
    """Returns prepared Titanic data.

    Args:
        df: Dataframe of raw Titanic data

    Returns:
        Prepared Titanic data.
    """
    # Lowercase columns
    df.columns = [col.lower() for col in df.columns]

    # Drop ticket and cabin col
    df.drop(columns=['passengerid', 'ticket', 'cabin'], inplace=True)

    # Create title column
    df = lowercase_column(df, 'name')
    df = extract_title(df, 'name', title_mapping)
    df.drop(columns=['name'], inplace=True)

    # Fill lowercase embarked column
    df = lowercase_column(df, 'embarked')
    df = lowercase_column(df, 'sex')

    # Fill nulls for numeric cols
    for col in ['pclass', 'age', 'sibsp', 'parch', 'fare']:
        df = fill_numeric(df, col, '-1')

    # Fill null values and categorical encoding
    df['title'].fillna('NA', inplace=True)
    df['sex'].fillna('NA', inplace=True)
    df['embarked'].fillna('NA', inplace=True)

    df['title'] = df['title'].map(title_to_int)
    df['sex'] = df['sex'].map(sex_to_int)
    df['embarked'] = df['embarked'].map(port_to_int)

    return df


def split_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits processed Titanic data into train and test dataframes.

    Args:
        df: Processed Titanic data

    Returns:
        Two dataframes, one for train and one for test.
    """
    train, test = train_test_split(df, test_size=0.2, random_state=1368, stratify=df['survived'])

    return train, test


def get_feats_and_labels(df: pd.DataFrame) -> Tuple[array, array]:
    """Returns a tuple of feature array and label vector

    Args:
        df: Processed Titanic data

    Returns:
        A tuple of feature array and label vector
    """
    # Get labels and features
    X = df.iloc[:, 1:].values
    y = df['survived'].values

    return X, y
