FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .

RUN echo "Building app..."

RUN pip install --no-cache-dir -r requirements.txt

RUN echo "Done with requirements..."

# Copy project files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "run:app", "--host", "0.0.0.0", "--port", "8000"]