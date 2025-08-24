
.PHONY: data train validate serve worker docker-build docker-up test

data:
	python -m edge.data.dataset_gen

train:
	python -m edge.train

validate:
	python -m edge.validate

serve:
	uvicorn edge.service.app:app --reload --host 0.0.0.0 --port 8000

worker:
	python -u edge/workers/policy_worker.py

docker-build:
	docker build -t edge-poc .

docker-up:
	docker-compose up --build

test:
	pytest -q
