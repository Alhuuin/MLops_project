FROM python:3.8-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-tk \
    tk \
    libffi-dev \ 
    libtk8.6 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

COPY . /app

RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]