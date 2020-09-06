import pandas as pd

from src.data_prep.prep_titanic import prep_df, get_feats_and_labels
from tests.tree.fixtures import dummy_titanic_dt, dummy_passengers, dummy_titanic


# Check if changing certain inputs will keep outputs constant
def test_dt_invariance(dummy_titanic_dt, dummy_passengers):
    model = dummy_titanic_dt
    p1, p2 = dummy_passengers

    # Get original survival probability of passenger 1
    test_df = pd.DataFrame.from_dict([p1], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p1_prob = model.predict(X)[0]  # 0.09

    # Change name from Owen to Mary (without changing gender or title)
    p1_name = p1.copy()
    p1_name['Name'] = ' Mr. Mary'
    test_df = pd.DataFrame.from_dict([p1_name], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p1_name_prob = model.predict(X)[0]  # 0.09

    # Change ticket number from 'A/5 21171' to 'PC 17599'
    p1_ticket = p1.copy()
    p1_ticket['ticket'] = 'PC 17599'
    test_df = pd.DataFrame.from_dict([p1_ticket], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p1_ticket_prob = model.predict(X)[0]  # 0.09

    # Change embarked port from 'S' to 'C'
    p1_port = p1.copy()
    p1_port['Embarked'] = 'C'
    test_df = pd.DataFrame.from_dict([p1_port], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p1_port_prob = model.predict(X)[0]  # 0.09

    # Get original survival probability of passenger 2
    test_df = pd.DataFrame.from_dict([p2], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p2_prob = model.predict(X)[0]  # 1.0

    # Change name from John to Berns (without changing gender or title)
    p2_name = p2.copy()
    p2_name['Name'] = ' Mrs. Berns'
    test_df = pd.DataFrame.from_dict([p2_name], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p2_name_prob = model.predict(X)[0]  # 1.0

    # Change ticket number from 'PC 17599' to 'A/5 21171'
    p2_ticket = p2.copy()
    p2_ticket['ticket'] = 'A/5 21171'
    test_df = pd.DataFrame.from_dict([p2_ticket], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p2_ticket_prob = model.predict(X)[0]  # 1.0

    # Change embarked port from 'C' to 'Q'
    p2_port = p2.copy()
    p2_port['Embarked'] = 'Q'
    test_df = pd.DataFrame.from_dict([p2_port], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p2_port_prob = model.predict(X)[0]  # 1.0

    assert p1_prob == p1_name_prob == p1_ticket_prob == p1_port_prob
    assert p2_prob == p2_name_prob == p2_ticket_prob == p2_port_prob


# Check if changing input (e.g., gender, passenger class) will affect survival probability in expected direction
def test_dt_directional_expectation(dummy_titanic_dt, dummy_passengers):
    model = dummy_titanic_dt
    p1, p2 = dummy_passengers

    # Get original survival probability of passenger 1
    test_df = pd.DataFrame.from_dict([p1], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p1_prob = model.predict(X)[0]  # 0.09

    # Change gender from male to female
    p1_female = p1.copy()
    p1_female['Name'] = ' Mrs. Owen'
    p1_female['Sex'] = 'female'
    test_df = pd.DataFrame.from_dict([p1_female], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p1_female_prob = model.predict(X)[0]  # 0.65

    # Change passenger class from 3 to 1
    p1_class = p1.copy()
    p1_class['Pclass'] = 1
    test_df = pd.DataFrame.from_dict([p1_class], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p1_class_prob = model.predict(X)[0]  # 0.36

    assert p1_prob < p1_female_prob, 'Changing gender from male to female should increase survival probability.'
    assert p1_prob < p1_class_prob, 'Changing class from 3 to 1 should increase survival probability.'

    # Get original survival probability of passenger 2
    test_df = pd.DataFrame.from_dict([p2], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p2_prob = model.predict(X)[0]  # 1.0

    # Change gender from female to male
    p2_male = p2.copy()
    p2_male['Name'] = ' Mr. John'
    p2_male['Sex'] = 'male'
    test_df = pd.DataFrame.from_dict([p2_male], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p2_male_prob = model.predict(X)[0]  # 0.56

    # Change class from 1 to 3
    p2_class = p2.copy()
    p2_class['Pclass'] = 3
    test_df = pd.DataFrame.from_dict([p2_class], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p2_class_prob = model.predict(X)[0]  # 0.0

    # Lower fare from 71.2833 to 5
    p2_fare = p2.copy()
    p2_fare['Fare'] = 5
    test_df = pd.DataFrame.from_dict([p2_fare], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p2_fare_prob = model.predict(X)[0]  # 0.85

    assert p2_prob > p2_male_prob, 'Changing gender from female to male should decrease survival probability.'
    assert p2_prob > p2_class_prob, 'Changing class from 1 to 3 should decrease survival probability.'
    assert p2_prob > p2_fare_prob, 'Changing fare from 72 to 5 should decrease survival probability.'
