FROM ubuntu:latest

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev git vim nano \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --no-cache-dir --upgrade pip

RUN mkdir -p /app
COPY requirements.txt /app

RUN cd /app & pip3 install --no-cache-dir -r requirements.txt