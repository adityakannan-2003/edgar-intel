.PHONY: help install dev up down psql ingest index eval eval-gate agent serve bench test lint fmt clean

SHELL := /bin/bash
PY    ?= python

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package plus dev extras
	$(PY) -m pip install -e ".[dev]"

dev:      ## Install everything including local models and serving
	$(PY) -m pip install -e ".[dev,models,serving,obs]"

up:       ## Start Postgres+pgvector and the OTel collector
	docker compose up -d db otel

down:     ## Stop everything
	docker compose down

psql:     ## Open a psql shell against the dev database
	docker compose exec db psql -U edgar -d edgar

ingest:   ## Fetch filings and XBRL facts for the configured company universe
	edgar-intel ingest run

index:    ## Chunk and embed under every strategy
	edgar-intel index build --strategy all

eval:     ## Run the full evaluation suite and record a run
	edgar-intel eval run --label "$(LABEL)"

eval-gate: ## Compare the latest run against the baseline and fail on regression
	edgar-intel eval gate --max-regression 0.03

agent:    ## Ask the bounded agent a question
	edgar-intel agent ask "$(Q)"

serve:    ## Run the FastAPI service locally
	uvicorn edgar_intel.serving.app:app --reload --port 8000

bench:    ## Latency / throughput / cost benchmark against the serving layer
	edgar-intel bench run --requests 200 --concurrency 8

test:     ## Run the test suite (no network, no model downloads)
	pytest

lint:
	ruff check src tests

fmt:
	ruff check --fix src tests
	ruff format src tests

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
