FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir .

EXPOSE 8000
VOLUME ["/app/desks", "/app/logs", "/app/outputs", "/app/zones"]
ENTRYPOINT ["employee-monitor"]
CMD ["run", "--headless", "--dashboard", "--config", "configs/config.example.yaml"]
