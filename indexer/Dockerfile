FROM python:3.11.6-slim as prod

ARG DEBIAN_FRONTEND=noninteractive

#install certificate
RUN apt-get update && apt install -y wget ca-certificates && apt-get clean
RUN wget -P /usr/local/share/ca-certificates/ https://gu-st.ru/content/Other/doc/russiantrustedca.pem && update-ca-certificates

COPY requirements.txt .
RUN pip install -r requirements.txt --trusted-host pypi.python.org --no-cache-dir

# Add our flow code to the image
WORKDIR /opt/prefect
COPY project project
COPY sources.yml sources.yml

# Run our flow script when the container starts
CMD ["python", "-m", "project.flow"]