FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application logic, static assets, and ML models
# (assuming the models like osis_snr_model.pkl are in the root directory)
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Start Uvicorn from main.py via app variable
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
