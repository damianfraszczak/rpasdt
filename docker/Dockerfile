FROM ubuntu:20.04

MAINTAINER damianfraszczak <damian.fraszczak@wat.edu.pl>

ENV NAME=rpasdt
ENV DEBIAN_FRONTEND=noninteractive
ENV LIBGL_ALWAYS_INDIRECT=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH "${PYTHONPATH}:/app/"

RUN adduser --quiet --disabled-password qtuser && usermod -a -G audio qtuser

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-pyqt5 \
    pyqt5-dev-tools \
    && cd /usr/local/bin \
    && ln -s /usr/bin/python3 python

RUN mkdir -p /app/rpasdt && chown -R qtuser /app/
WORKDIR /app/
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY docker/docker-entrypoint.sh docker-entrypoint.sh
RUN chmod +x docker-entrypoint.sh

USER qtuser
COPY src/rpasdt /app/rpasdt
COPY pyinstaller_config /app/pyinstaller_config
ENTRYPOINT ["/app/docker-entrypoint.sh"]
