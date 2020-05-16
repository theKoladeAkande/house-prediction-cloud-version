FROM python:3.7.3

LABEL "Kolade"="akandekolade@gmail.com"
LABEL version="0.1.0"
LABEL description="Custom regression model for sageMaker"

RUN apt-get -y update && apt-get install -y --no-install-recommends \
    nginx\
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY xgboost /opt/program

WORKDIR /opt/program

RUN pip install -r /opt/program/requirements.txt

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="$PATH:/opt/program"






