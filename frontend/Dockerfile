# frontend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy ONLY the frontend requirements
COPY frontend/requirements.txt .

# Install the lightweight dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the frontend code
COPY frontend/ .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]