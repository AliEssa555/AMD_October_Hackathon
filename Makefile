# YOLO-DER task runner.  `make help` lists targets.
# Windows (no make): use ./tasks.ps1 <target> -- same target names.
.DEFAULT_GOAL := help
SHELL := /bin/bash
PY ?= python

-include .env
export

.PHONY: help setup lint fmt test test-all data smoke tunnel launch reap status export api-dev clean

help:  ## list targets
	@grep -hE '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:  ## create .venv and install the package with [dev] + [train] extras
	$(PY) -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -e ".[dev,train]" && pre-commit install

lint:  ## ruff check + format check
	ruff check .
	ruff format --check .

fmt:  ## ruff autofix + format
	ruff check --fix .
	ruff format .

test:  ## fast unit tests (CI gate)
	pytest -m "not slow"

test-all:  ## every test, including slow ones
	pytest

data:  ## run the DVC data pipeline (download -> prepare -> severity)
	dvc repro

smoke:  ## 1-epoch run on 2% of the data (CPU / Colab T4 sanity check)
	$(PY) -m yolo_der.train --config configs/baseline.yaml --seed 0 --fraction 0.02 --epochs-override 1

tunnel:  ## SSH tunnel: localhost:5000 -> hub MLflow over the VPC
	ssh -N -L 5000:$(HUB_PRIVATE_IP):5000 root@$(HUB_PUBLIC_IP)

launch:  ## create an ephemeral MI300X Droplet (asks for confirmation + prints cost)
	bash infra/launch_gpu.sh $(ARGS)

reap:  ## destroy any ephemeral-gpu Droplet older than REAPER_MAX_AGE_HOURS
	bash infra/reaper.sh

status:  ## list every yolo-der cloud resource + remaining credit
	bash infra/status.sh

export:  ## export the champion run to ONNX and upload to Spaces
	$(PY) -m yolo_der.export $(ARGS)

api-dev:  ## run the Node inference API in watch mode
	cd api && npm run dev

clean:  ## remove caches and local run output
	rm -rf .pytest_cache .ruff_cache **/__pycache__ runs/*.tmp
