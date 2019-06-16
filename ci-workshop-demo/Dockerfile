FROM ci-workshop-demo-base:latest

RUN mkdir -p /app/continuous-intelligence/{src,data}

COPY start.sh /app/continuous-intelligence
COPY src /app/continuous-intelligence/src
COPY data/models /app/continuous-intelligence/data/models

RUN chmod +x /app/continuous-intelligence/start.sh

EXPOSE 5005

CMD ["/app/continuous-intelligence/start.sh"]
