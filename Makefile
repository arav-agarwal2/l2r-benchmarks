# docstr-cov darglint sphinx for documentation
# mypy for type coverage
# pytest-cov for test coverage
# black
test:
  python -m pytest --cov-report html --cov=src tests/*/*

