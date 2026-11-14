.PHONY: install test test-unit test-integration pack status

install:
	python3.12 -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/pytest -q

test-unit:
	.venv/bin/pytest -q -m "not integration"

test-integration:
	.venv/bin/pytest -q -m integration

pack:
	folio pack

status:
	folio status
