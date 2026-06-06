.PHONY: help venv install freeze eda test lab validate clean

PY := .venv/bin/python
PIP := .venv/bin/uv pip

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

venv:           ## create the uv venv (python 3.12)
	uv venv --python 3.12 .venv

install:        ## install pinned dev deps into the venv
	$(PIP) install -r requirements-dev.txt

freeze:         ## refreeze the lockfile from the current venv
	.venv/bin/uv pip freeze > requirements-dev.txt

eda:            ## execute the EDA notebook end-to-end
	.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

test:           ## run the pytest suite
	$(PY) -m pytest -q tests/

lab:            ## launch jupyter lab
	.venv/bin/jupyter lab

validate:       ## run the official submission validator (pass SUB=path/to/folder)
	$(PY) validate_submission.py $(SUB)

shap:           ## compute SHAP driver explanation (reports/figures/shap_summary.png)
	$(PY) -m src.explain

pdf:            ## compile Deliverable D writeup (LaTeX) to PDF
	cp submissions/submission_D_writeup.tex /tmp/
	cp reports/figures/shap_summary.png reports/figures/fig01_prior_score_threshold.png reports/figures/fig03_cumulative_trajectory.png reports/figures/proxy_structure.png /tmp/
	cd /tmp && pdflatex -interaction=nonstopmode submission_D_writeup.tex >/dev/null 2>&1 \
	  && pdflatex -interaction=nonstopmode submission_D_writeup.tex >/dev/null 2>&1
	cp /tmp/submission_D_writeup.pdf submissions/submission_D_writeup.pdf
	@echo "wrote submissions/submission_D_writeup.pdf"

clean:          ## remove caches
	rm -rf .pytest_cache **/__pycache__ artifacts/tmp
