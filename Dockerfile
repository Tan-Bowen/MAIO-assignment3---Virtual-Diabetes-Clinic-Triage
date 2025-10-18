# Stage 1: Builder
# Use a slim-buster image for a smaller base build image
FROM python:3.11-slim-buster AS builder

# Set environment variables
ENV PYTHONUNBUFFERED 1

# Install necessary system dependencies for building
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY app/ ./app/

# Train model (v0.1 baseline) and place it in the working directory
# Note: The version here is the default, overridden by build-args in CI/CD
ARG MODEL_VERSION=v0.1
ENV MODEL_VERSION=${MODEL_VERSION}
RUN python app/train.py

# --- Stage 2: Runner ---
# Use a minimal base image for the final runtime
FROM python:3.11-slim-buster AS runner

# Set working directory
WORKDIR /app

# Copy only the necessary runtime files from the builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app/app/ ./app/
COPY --from=builder /app/model.joblib .

# Set environment variables for the service
ENV MODEL_VERSION=v0.1
ENV PORT=8000
ENV HOST=0.0.0.0

# Expose port
EXPOSE 8000

# Set the entrypoint to run the FastAPI app
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]