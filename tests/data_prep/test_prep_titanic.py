from src.data_prep.prep_titanic import load_df, prep_df, split_df, get_feats_and_labels


# Assert data set shape
def test_prep_steps():
    df = load_df()
    df = prep_df(df)

    train, test = split_df(df)
    X_train, y_train = get_feats_and_labels(train)
    X_test, y_test = get_feats_and_labels(test)

    assert X_train.shape == (712, 8)
    assert y_train.shape == (712,)
    assert X_test.shape == (179, 8)
    assert y_test.shape == (179,)
