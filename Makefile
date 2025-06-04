# 📦 Makefile for stance classification pipeline

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
JUPYTER = $(VENV)/bin/jupyter

all: eda train report

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

eda:
	$(JUPYTER) nbconvert --to notebook --execute eda_stance_classification.ipynb --output eda_output.ipynb

train:
	$(JUPYTER) nbconvert --to notebook --execute train_stance_kfold_bert.ipynb --output training_output.ipynb

report:
	$(JUPYTER) nbconvert --to notebook --execute report_stance_classification.ipynb --output report_output.ipynb

clean:
	rm -rf __pycache__ output/ eda_plots/ *.ipynb_checkpoints *.csv *.png *.pdf .venv

.PHONY: all setup eda train report clean
