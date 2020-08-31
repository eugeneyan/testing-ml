# testing-ml


Examples on how to test machine learning code. We'll test a `numpy` implementation of DecisionTree and RandomForest, covering some [standard software tests](https://github.com/eugeneyan/testing-ml#standard-software-tests), [model tests](https://github.com/eugeneyan/testing-ml#model-tests), and [model evaluation](https://github.com/eugeneyan/testing-ml#model-evaluation).

Inspired by [@jeremyjordan](https://twitter.com/jeremyjordan)'s [Effective Testing for Machine Learning Systems](https://www.jeremyjordan.me/testing-ml/); follow-up article on 2020-09-06 @ [eugeneyan.com](https://eugeneyan.com/writing/).

![Tests](https://github.com/eugeneyan/testing-ml/workflows/Tests/badge.svg?branch=master) [![codecov](https://codecov.io/gh/eugeneyan/testing-ml/branch/master/graph/badge.svg)](https://codecov.io/gh/eugeneyan/testing-ml) [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/eugeneyan/testing-ml/pulls)

## Quick Start
```
# Clone and setup environment
git clone git@github.com:eugeneyan/testing-ml.git
cd testing-ml
make setup

# Run test suite
make check
```

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
