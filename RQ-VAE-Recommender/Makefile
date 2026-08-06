CONDA_ENV ?= NewsRecommend
PYTHON ?= conda run -n $(CONDA_ENV) env PYTHONPATH=. PYTHONUNBUFFERED=1 python -u

help: ## Show help
	grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

env_smoke: ## Verify core dependencies in the target conda env
	$(PYTHON) scripts/env_smoke.py

train_rqvae: ## Train RQ-VAE with gin config=...
	$(PYTHON) train_rqvae.py $(config)

export_bridge: ## Export bridge artifact from checkpoint (optional overrides: checkpoint=..., export_path=...)
	$(PYTHON) scripts/export_bridge_from_checkpoint.py --checkpoint_path $(or $(checkpoint),out/rqvae/ml1m/checkpoint_9999.pt) --export_path $(or $(export_path),out/rqvae/ml1m/bridge_artifacts.pt)

export_dense: ## Export dense MiniLM item features artifact for HSTU dense baseline
	$(PYTHON) scripts/export_dense_features.py --dataset_folder $(or $(dataset_folder),dataset/ml-1m) --output_path $(or $(output_path),out/rqvae/ml1m/dense_features.pt) --bridge_path $(or $(bridge_path),out/rqvae/ml1m/bridge_artifacts.pt)

audit_rqvae: ## Audit RQ-VAE training quality and bridge health
	$(PYTHON) scripts/audit_rqvae_quality.py --metrics_csv_path $(or $(metrics_csv_path),out/rqvae/ml1m/metrics.csv) --bridge_path $(or $(bridge_path),out/rqvae/ml1m/bridge_artifacts.pt) $(if $(cold_items_path),--cold_items_path $(cold_items_path),) $(if $(output_json),--output_json $(output_json),)

train_decoder: ## Train semantic decoder with gin config=...
	$(PYTHON) train_decoder.py $(config)
