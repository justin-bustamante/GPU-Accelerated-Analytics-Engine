PY ?= python
DATA ?= data

.PHONY: install install-cu12 install-cu13 lint test test-gpu check-gpu sample dev-data clean-data

install:            ## CPU-only environment
	$(PY) -m pip install -e ".[dev]"

install-cu12:       ## GPU environment, driver reports CUDA 12.x
	$(PY) -m pip install -e ".[dev,gpu-cu12]"

install-cu13:       ## GPU environment, driver reports CUDA 13.x
	$(PY) -m pip install -e ".[dev,gpu-cu13]"

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

test:
	$(PY) -m pytest

test-gpu:
	$(PY) -m pytest -m gpu

check-gpu:          ## full environment validation, exits non-zero if the GPU path is unusable
	$(PY) -m engine inspect-gpu --smoke --require-gpu

sample:             ## 100K rows, a few MB, seconds to generate
	$(PY) -m engine generate --rows 100K --output $(DATA)/events-100k --overwrite
	$(PY) -m engine inspect --dataset $(DATA)/events-100k

dev-data: sample    ## the development ladder: 100K, 1M, 10M
	$(PY) -m engine generate --rows 1M --output $(DATA)/events-1m --overwrite --workers 4
	$(PY) -m engine generate --rows 10M --output $(DATA)/events-10m --overwrite --workers 4

clean-data:
	rm -rf $(DATA)/events-100k $(DATA)/events-1m $(DATA)/events-10m
