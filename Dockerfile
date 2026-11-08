FROM apache/airflow:3.4.0-python3.12

# Install the pipeline's libraries on top of Airflow, pinned to Airflow's own
# constraints so pip can't upgrade anything Airflow depends on.
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir "apache-airflow==3.4.0" -r /requirements.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.4.0/constraints-3.12.txt"
