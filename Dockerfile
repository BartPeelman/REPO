# Use Python 3.12 slim image for a smaller footprint
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements.txt first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model repository
COPY model_repository ./model_repository

# Copy the Flask application code
COPY app.py .

# Expose Flask port
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "app.py"]
