FROM ubuntu:latest

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev git vim nano \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --no-cache-dir --upgrade pip

RUN mkdir -p /app/continuous-intelligence \
  && git clone https://github.com/stayrascal/continuous-intelligence.git /app/continuous-intelligence \
  && cd /app/continuous-intelligence/ci-workshop-demo \
  && pip3 install --no-cache-dir -r requirements.txt


WORKDIR /app/continuous-intelligence/ci-workshop-demo