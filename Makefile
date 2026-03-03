.PHONY: setup ingest features train eval api test all

setup:
	pip install -r requirements.txt

ingest:
	python src/ingest.py

features:
	python src/features.py

train:
	python src/train_classifier.py
	python src/train_recommender.py

eval:
	python src/evaluate.py

api:
	uvicorn src.api:app --reload

test:
	pytest tests/

all: setup ingest features train eval
