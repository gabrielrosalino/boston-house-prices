.PHONY: install install-dev lint format type-check test train api docker-build docker-run

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,api,notebook]"

lint:
	ruff check .

format:
	ruff format .
	ruff check . --fix

type-check:
	mypy src

test:
	pytest

train:
	boston-train

api:
	uvicorn boston_house_prices.api:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t boston-house-prices:latest .

docker-run:
	docker run --rm -p 8000:8000 -v $(PWD)/models:/app/models boston-house-prices:latest
