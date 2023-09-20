FROM apache/airflow:2.3.0

RUN pip install geopandas
RUN pip install gdown
RUN pip install keplergl