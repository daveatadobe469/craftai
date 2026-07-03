.PHONY: dev api ui test index lint clean help

PYTHON  := python3
UVICORN := uvicorn
STREAMLIT := streamlit
PYTEST  := pytest
RUFF    := ruff
MYPY    := mypy

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

dev: ## Start FastAPI + Streamlit concurrently (production-like)
	@echo "🚀 Starting CRAFTAI…"
	@echo "   API  → http://localhost:8000"
	@echo "   UI   → http://localhost:8501"
	@echo "   Docs → http://localhost:8000/docs"
	@$(UVICORN) api.main:app --host 0.0.0.0 --port 8000 --reload & \
	$(STREAMLIT) run ui/app.py --server.port 8501 --server.address 0.0.0.0; \
	wait

api: ## Start only the FastAPI backend
	$(UVICORN) api.main:app --host 0.0.0.0 --port 8000 --reload

ui: ## Start only the Streamlit UI
	$(STREAMLIT) run ui/app.py --server.port 8501 --server.address 0.0.0.0

index: ## Seed ChromaDB collections and brand guidelines
	@echo "📦 Initialising ChromaDB collections…"
	$(PYTHON) scripts/init_chroma.py
	@echo "📋 Seeding brand guidelines…"
	$(PYTHON) scripts/generate_guidelines.py
	@echo "✅ Knowledge base ready."

cluster: ## Cluster persona embeddings and generate report
	$(PYTHON) scripts/cluster_personas.py

test: ## Run all tests with pytest
	$(PYTEST) tests/ -v --tb=short --asyncio-mode=auto

test-compliance: ## Run compliance unit tests only
	$(PYTEST) tests/test_compliance.py -v

test-ragas: ## Run RAGAS evaluation tests (requires GROQ_API_KEY)
	$(PYTEST) tests/test_ragas.py -v --asyncio-mode=auto

test-graph: ## Run graph integration tests
	$(PYTEST) tests/test_graph.py -v --asyncio-mode=auto

lint: ## Run ruff linter + mypy type checker
	$(RUFF) check . --fix
	$(MYPY) config.py graph/ api/ rag/ compliance/ db/ --ignore-missing-imports

clean: ## Remove generated data directories
	rm -rf data/chroma data/mlruns data/craftai.db data/cluster_report.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

setup: ## Copy .env.example to .env (run once)
	@if [ ! -f .env ]; then cp .env.example .env; echo "✅ .env created from .env.example"; \
	else echo "⚠️  .env already exists — not overwriting."; fi

init: setup index ## Full first-time setup: copy .env + seed KB
	@echo "✅ CRAFTAI ready. Run 'make dev' to start."
