COMPOSE ?= docker compose
DAG := citibike_weather_pipeline

.PHONY: help up down logs test test-agents sample-data trigger clean agents agents-down part3-eval part3-run results

help:
	@echo "make up           build images and start Airflow (UI: http://localhost:8080, airflow/airflow)"
	@echo "make trigger      unpause the DAG and trigger a run"
	@echo "make logs         follow scheduler logs"
	@echo "make test         Part 1 unit tests (inside the Airflow container)"
	@echo "make test-agents  Part 2 + 3 unit tests (inside a Python 3.12 container)"
	@echo "make results      copy results to docs/results/ and write them into the READMEs"
	@echo "make sample-data  write synthetic raw files to data/raw (offline demo)"
	@echo "make down         stop containers (databases are kept)"
	@echo "make clean        stop and delete databases, logs and data"
	@echo "make agents       Part 2: start the agent lab (UI: http://127.0.0.1:5173/solution-loop)"
	@echo "make agents-down  Part 2: stop the agent lab"
	@echo "make part3-eval   Part 3: evaluation suite -> agent-lab/adaptive/eval/out/dashboard.html"
	@echo "make part3-run    Part 3: adaptive system on the real Citi Bike data"

up:
	$(COMPOSE) up -d --build
	@echo "Airflow UI: http://localhost:8080  (user airflow / password airflow)"

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f airflow-scheduler

test:
	$(COMPOSE) run --rm --entrypoint python dev -m pytest -q tests

sample-data:
	$(COMPOSE) run --rm --entrypoint python dev scripts/make_sample_data.py \
		--months 2026-04 2026-05 2026-06 2026-07 2026-08 2026-09 --out data/raw

trigger:
	$(COMPOSE) exec airflow-scheduler airflow dags unpause $(DAG)
	$(COMPOSE) exec airflow-scheduler airflow dags trigger $(DAG)

clean:
	$(COMPOSE) down -v
	rm -rf data/raw data/processed data/output logs/*/

agents:
	$(MAKE) -C agent-lab start-mac

agents-down:
	$(MAKE) -C agent-lab down-mac

part3-eval:
	$(MAKE) -C agent-lab part3-eval

part3-run:
	$(MAKE) -C agent-lab part3-run FAULT=$(FAULT)

test-agents:
	$(MAKE) -C agent-lab test-agents

results:
	python3 scripts/update_results.py
