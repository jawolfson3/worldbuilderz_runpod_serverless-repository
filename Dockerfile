# Use an official Python runtime
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Set the default command (this runs when container starts)
CMD ["python", "src/inference.py"]
