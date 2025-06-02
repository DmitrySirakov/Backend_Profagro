FROM python:3.11.6-slim as prod

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONPATH /opt/application/
ENV PATH /opt/application/:$PATH
ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/application/

COPY requirements.txt /opt/application/
RUN pip install -r requirements.txt
RUN pip install --upgrade certifi

RUN apt-get update && apt install -y wget ca-certificates && apt-get clean
RUN wget -P /usr/local/share/ca-certificates/ https://gu-st.ru/content/Other/doc/russiantrustedca.pem && update-ca-certificates

# use non-root user
RUN useradd -g users user

COPY --chown=user:users project /opt/application/project

ENTRYPOINT uvicorn project.app:app --host $HOST --port $PORT --workers $WORKERS