FROM python:3.11-slim

WORKDIR /app

# Install CPU-only torch first to keep the image lean
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic

COPY backend ./backend
COPY frontend ./frontend

EXPOSE 8000
# Respect $PORT (DigitalOcean App Platform, Render, etc.); default to 8000 locally.
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
