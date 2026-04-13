# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for FAISS and compiling Python packages
RUN apt-get update && \
    apt-get install -y build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# --- STEP 5 FIX ---
# Create the data directory and give it open read/write/execute permissions.
# This prevents PermissionError crashes when Docker maps the host volume.
RUN mkdir -p /app/data && chmod 777 /app/data

# Expose the FastAPI port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "10000"]