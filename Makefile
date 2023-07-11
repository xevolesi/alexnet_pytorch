SHELL := /bin/bash

.EXPORT_ALL_VARIABLES:
PYTHONPATH := ./
TEST_DIR := tests/
LINT_DIR := ./
MYPY_DIR := source


format:
	ruff check ${LINT_DIR} --fix

lint:
	ruff check ${LINT_DIR}

run_tests:
	pytest -svvv ${TEST_DIR}

type_check:
	mypy -p ${MYPY_DIR}

# Call this before commit.
pre_push_test: type_check lint run_tests
