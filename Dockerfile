# Multi-stage build for smaller image size
FROM python:3.11-slim AS builder

WORKDIR /flask_app

# Install only essential build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY app/requirements.txt /flask_app/app/requirements.txt
RUN pip install --no-cache-dir --user -r app/requirements.txt

# Final stage - minimal runtime image
FROM python:3.11-slim

WORKDIR /flask_app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY app/ /flask_app/app/

# Add .local/bin to PATH
ENV PATH=/root/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app.app:app"]


