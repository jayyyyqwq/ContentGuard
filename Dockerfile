FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY . .

RUN pip install --no-cache-dir -e ".[core]" 2>/dev/null || pip install --no-cache-dir \
    "openenv-core[core]>=0.2.2" \
    "openai>=1.0.0" \
    "uvicorn[standard]>=0.24.0"

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
