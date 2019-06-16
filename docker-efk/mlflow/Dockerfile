FROM continuumio/miniconda:4.5.4
ENV MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT "/tmp/artifact/"
ENV AZURE_STORAGE_ACCESS_KEY ""
ENV GCS_STORAGE_ACCESS_KEY ""


WORKDIR /
COPY . /

RUN pip install mlflow==0.9.1 \
    && pip install azure-storage==0.36.0 \
    && pip install google-cloud-storage==1.16.0 \
    && pip install numpy==1.14.3 \
    && pip install pandas==0.22.0 \
    && pip install scikit-learn==0.19.1 \
    && pip install cloudpickle

ENTRYPOINT ["./startup.sh"]