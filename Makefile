COMPOSE ?= docker compose
DAG := citibike_weather_pipeline

.PHONY: help up down logs test sample-data trigger clean

help:
	@echo "make up           build images and start Airflow (UI: http://localhost:8080, airflow/airflow)"
	@echo "make trigger      unpause the DAG and trigger a run"
	@echo "make logs         follow scheduler logs"
	@echo "make test         run the unit tests inside the container"
	@echo "make sample-data  write synthetic raw files to data/raw (offline demo)"
	@echo "make down         stop containers (databases are kept)"
	@echo "make clean        stop and delete databases, logs and data"

up:
	$(COMPOSE) up -d --build
	@echo "Airflow UI: http://localhost:8080  (user airflow / password airflow)"

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f airflow-scheduler

test:
	$(COMPOSE) run --rm dev pytest -q tests

sample-data:
	$(COMPOSE) run --rm dev python scripts/make_sample_data.py \
		--months 2026-04 2026-05 2026-06 2026-07 2026-08 2026-09 --out data/raw

trigger:
	$(COMPOSE) exec airflow-scheduler airflow dags unpause $(DAG)
	$(COMPOSE) exec airflow-scheduler airflow dags trigger $(DAG)

clean:
	$(COMPOSE) down -v
	rm -rf data/raw data/processed data/output logs/*/
