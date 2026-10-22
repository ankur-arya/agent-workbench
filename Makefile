PYTHON ?= python
HOST ?= 127.0.0.1
PORT ?= 8000
export PYTHONPATH := $(CURDIR)

.PHONY: help install install-dev frontend demo eval info ui test clean

help:
	@echo "Agent workbench — build, preview, and evaluate agents (no Docker)"
	@echo ""
	@echo "  make install     Create .venv and install Python deps"
	@echo "  make frontend    npm install + build the SPA"
	@echo "  make ui          Build UI if needed and serve API+UI on :$(PORT)"
	@echo "  make demo        Run the tool-assistant sample"
	@echo "  make eval        Grade the built-in template tasks"
	@echo "  make test        pytest"
	@echo "  make clean       Remove local store, caches, frontend dist"

install:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

install-dev: install
	.venv/bin/pip install -e .

frontend:
	cd frontend && npm install && npm run build

demo:
	$(PYTHON) -m workbench demo

eval:
	$(PYTHON) -m workbench eval --template tool_assistant
	$(PYTHON) -m workbench eval --template research
	$(PYTHON) -m workbench eval --template triage
	$(PYTHON) -m workbench eval --template guardrailed

info:
	$(PYTHON) -m workbench info

ui:
	$(PYTHON) -m workbench ui --host $(HOST) --port $(PORT)

test:
	$(PYTHON) -m pytest

clean:
	rm -rf .workbench .pytest_cache frontend/dist frontend/node_modules
	find . -type d -name __pycache__ -exec rm -rf {} +
