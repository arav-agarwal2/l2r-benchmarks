.PHONY: test black docstr_cov typecheck_cov check docs
test:
	python -m pytest --cov-report html --cov=src tests/*/*

black:
	python -m black src/ tests/ docs/source/ scripts/

docstr_cov:
	docstr-coverage src/
	darglist src/

typecheck_cov:
	mypy src/ --ignore-missing-imports --strict

docs: 
	sphinx-apidoc -f -o docs/source/ src
	cd docs && make html && cd ..

check: test black docstr_cov typecheck_cov
	pass
# sphinx for documentation
# mypy for type coverage