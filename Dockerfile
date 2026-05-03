# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
# Adding fastapi, uvicorn, and python-multipart explicitly as they are required for the API
# Increased timeout to 1000s to handle large packages on slow networks
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt fastapi uvicorn python-multipart

# Copy project
COPY . .

# Create directories for outputs and state if they don't exist
RUN mkdir -p outputs agent_state

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
