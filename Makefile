.PHONY: test black docstr_cov typecheck_cov check
test:
	python -m pytest --cov-report html --cov=src tests/*/*

black:
	python -m black src/ tests/

docstr_cov:
	docstr-coverage src/
	darglist src/

typecheck_cov:
	mypy src/ --ignore-missing-imports --strict

check: test black docstr_cov typecheck_cov

# sphinx for documentation
# mypy for type coverage