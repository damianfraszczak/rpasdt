FROM ubuntu:20.04


ENV NAME=rpasdt
ENV DEBIAN_FRONTEND=noninteractive
ENV LIBGL_ALWAYS_INDIRECT=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH "${PYTHONPATH}:/app/"
ENV MPLBACKEND=Qt5Agg


RUN adduser --quiet --disabled-password qtuser && usermod -a -G audio qtuser

RUN apt-get update && \
    apt-get install -y \
    qt5-default \
    python3-pip \
    python3-pyqt5 \
    pyqt5-dev-tools \
    && cd /usr/local/bin \
    && ln -s /usr/bin/python3 python


# python3-graph-tool config
#RUN  echo "deb https://downloads.skewed.de/apt focal main" >> /etc/apt/sources.list && \
#     apt-key adv --keyserver keyserver.ubuntu.com --recv-key 612DEFB798507F25 && \
#     apt-get update && apt-get install -y  python3-graph-tool

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
