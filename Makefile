.PHONY: install lint typecheck test check

install:
	python -m pip install -e ".[dev,asr]"

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy

test:
	pytest

check: lint typecheck test
