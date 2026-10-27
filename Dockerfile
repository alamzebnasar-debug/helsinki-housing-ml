# Use a lightweight official Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy and install dependencies first (leverages Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project structure into the container
COPY . .

# Expose the port that FastAPI runs on
EXPOSE 8000

# Run Uvicorn, pointing to your API module inside the api/ folder
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
