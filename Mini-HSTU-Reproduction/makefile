
CONDA_ENV ?= NewsRecommend
PYTHON ?= conda run -n $(CONDA_ENV) env PYTHONPATH=src python

help:  ## Show help
	grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

format: ## Run pre-commit hooks to format the code
	conda run -n $(CONDA_ENV) pre-commit run -a

test: ## Run all tests
	$(PYTHON) -m coverage erase
	$(PYTHON) -m coverage run --source=src/ -m pytest tests --durations=10 -vv
	$(PYTHON) -m coverage report --format=markdown

train: ## Train the model
	$(PYTHON) src/generative_recommenders_pl/scripts/train.py $(MAKEOVERRIDES)

eval: ## Evaluate the model
	$(PYTHON) src/generative_recommenders_pl/scripts/eval.py $(MAKEOVERRIDES)

predict: ## Predict the model
	$(PYTHON) src/generative_recommenders_pl/scripts/predict.py $(MAKEOVERRIDES)

prepare_data: ## Prepare data
	$(PYTHON) src/generative_recommenders_pl/scripts/prepare_data.py $(MAKEOVERRIDES)

env_smoke: ## Verify core dependencies in the target conda env
	$(PYTHON) scripts/env_smoke.py

summarize_ablation: ## Summarize cold-start ablation runs into docs tables (baseline/dense/routeA/routeB)
	$(PYTHON) scripts/summarize_ablation.py --baseline_run $(baseline_run) $(if $(dense_run),--dense_run $(dense_run),) --route_a_run $(route_a_run) --route_b_run $(route_b_run) --output_markdown $(or $(output_markdown),docs/ablation_results_ml1m.md) --output_csv $(or $(output_csv),docs/ablation_results_ml1m.csv)

full_token_smoke: ## Run the synthetic CPU correctness smoke (never full training)
	$(PYTHON) scripts/full_token_smoke.py

summarize_full_token: ## Summarize one completed Full-Token Route B server run
	$(PYTHON) scripts/summarize_full_token_route_b.py --run_dir $(run_dir) --prior_report ../reproduction/FINAL_REPORT.md --output_results ../reproduction/full_token_route_b_results.md --output_report ../reproduction/FULL_TOKEN_ROUTE_B_REPORT.md
