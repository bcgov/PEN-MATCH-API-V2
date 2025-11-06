FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml requirements.txt* ./

# Install all dependencies
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    scikit-learn>=1.4.0 \
    numpy>=1.26.0 \
    pandas \
    requests==2.31.0 \
    python-dotenv==1.0.0 \
    pydantic==2.5.0 \
    pydantic-settings \
    python-multipart \
    openai \
    azure-identity \
    azure-core \
    azure-cosmos==4.5.1 \
    azure-keyvault-secrets \
    azure-storage-blob \
    azure-search-documents \
    azure-ai-documentintelligence \
    beautifulsoup4 \
    structlog \
    psutil \
    httpx \
    langchain \
    langchain-openai \
    langchain-community \
    langgraph \
    tavily-python \
    redis \
    pytest==7.4.3 \
    pytest-asyncio==0.21.1

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p app/logs

# Expose port 8000 for FastAPI
EXPOSE 8000

# Set environment variables for Azure App Service
ENV PORT=8000
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1

# Use the PORT environment variable that Azure provides
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]