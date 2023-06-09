SHELL := /bin/bash

.EXPORT_ALL_VARIABLES:
PYTHONPATH := ./
TEST_DIR := tests/
LINT_DIR := ./


format:
	ruff check ${LINT_DIR} --fix

lint:
	ruff check ${LINT_DIR}

run_tests:
	pytest -svvv ${TEST_DIR}

# Call this before commit.
pre_push_test: lint run_tests
