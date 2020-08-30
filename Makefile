setup:
	pip install poetry
	poetry install

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -f .coverage
	rm -f .coverage.*

clean: clean-pyc clean-test

test: clean
	poetry run py.test tests --cov-config=.coveragerc --cov=src

mypy:
	poetry run mypy src

lint:
	poetry run pylint src -j 4 --reports=y

check: test lint mypy