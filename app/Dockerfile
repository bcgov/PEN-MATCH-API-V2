FROM python:3.13-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv (dependency manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv or fallback to pip
RUN uv sync --frozen --no-cache --no-dev || \
    pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    scikit-learn==1.3.2 \
    numpy==1.24.3 \
    requests==2.31.0 \
    azure-cosmos==4.5.1 \
    python-dotenv==1.0.0 \
    pydantic==2.5.0

# Copy application code
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Set environment variables for Azure App Service
ENV PORT=8000
ENV HOST=0.0.0.0

# Health check endpoint for Azure App Service
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

# Run the FastAPI app using uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]