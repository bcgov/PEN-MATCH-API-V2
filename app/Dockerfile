# Use Python slim base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app


RUN apt-get update && apt-get install -y \
    libaio1t64 \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Oracle Instant Client (current 23.26 basiclite)

RUN curl -L -o instantclient-basiclite.zip \
    https://download.oracle.com/otn_software/linux/instantclient/2326000/instantclient-basiclite-linux.x64-23.26.0.0.0.zip \
    && unzip instantclient-basiclite.zip -d /opt/oracle \
    && rm instantclient-basiclite.zip

    

# Set Oracle environment variable
ENV LD_LIBRARY_PATH=/opt/oracle/instantclient_23_6

# Copy dependency file FIRST (docker caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire app source
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Default production command (100% works on Azure App Service)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
