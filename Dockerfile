# Assume we are starting with the Dockerfile setup for your application
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt first for better cache utilization
COPY requirements.txt .
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the SpaCy model
RUN python -m spacy download en_core_web_sm

# Copy the rest of your application
COPY . .

# Command to run your application
CMD ["python", "./model_training.py"]
