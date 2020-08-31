# testing-ml

Examples on how to test machine learning systems. Inspired by this [article](https://www.jeremyjordan.me/testing-ml/); accompanying article coming 2020-09-06.

![Tests](https://github.com/eugeneyan/testing-ml/workflows/Tests/badge.svg?branch=master) [![codecov](https://codecov.io/gh/eugeneyan/testing-ml/branch/master/graph/badge.svg)](https://codecov.io/gh/eugeneyan/testing-ml)

## Standard software tests
- [Unit tests & Code coverage](https://github.com/eugeneyan/testing-ml/blob/master/Makefile#L17)
- [Linting](https://github.com/eugeneyan/testing-ml/blob/master/Makefile#L23)
- [Type checking](https://github.com/eugeneyan/testing-ml/blob/master/Makefile#L20)


## Model tests
- Pre-train tests
	- [Check on test dataset shape](https://github.com/eugeneyan/testing-ml/blob/master/tests/data_prep/test_prep_titanic.py#L5)
	- [Checking output shape and range](https://github.com/eugeneyan/testing-ml/blob/master/tests/tree/test_decision_tree.py#L91)
	- [Check if if model can overfit perfectly](https://github.com/eugeneyan/testing-ml/blob/master/tests/tree/test_decision_tree.py#L114)
	- [Check if additional DecisionTree depth increases training accuracy](https://github.com/eugeneyan/testing-ml/blob/master/tests/tree/test_decision_tree.py#L136)
	- [Check if additional RandomForest trees increases validation accuracy](https://github.com/eugeneyan/testing-ml/blob/master/tests/tree/test_random_forest.py#L27)

- Post-train tests
	- [Check invariance (i.e., change in input doesn't affect output)](https://github.com/eugeneyan/testing-ml/blob/master/tests/tree/test_decision_tree.py#L154)
	- [Check directional expectation (i.e., output changes in an expected manner)](https://github.com/eugeneyan/testing-ml/blob/master/tests/tree/test_decision_tree.py#L215)
	- [Check minimum functionality (e.g., with null values)](https://github.com/eugeneyan/testing-ml/blob/master/tests/tree/test_decision_tree.py#L276)
	- [Check Random Forest outperforms Decision Trees](https://github.com/eugeneyan/testing-ml/blob/master/tests/tree/test_random_forest.py#L45)

	
## Model evaluation
- Evaluation on train-test split
	- [Decision Tree Accuracy and AUC ROC](https://github.com/eugeneyan/testing-ml/blob/master/tests/tree/test_decision_tree.py#L325)
	- [Random Forest Accuracy and AUC ROC](https://github.com/eugeneyan/testing-ml/blob/master/tests/tree/test_random_forest.py#L68)
