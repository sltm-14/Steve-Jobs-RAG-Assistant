FROM python:3.11-slim

WORKDIR /home/app

COPY requirements.txt .

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY . /home/app    

# Start the FastAPI application using Uvicorn.
# Format: uvicorn <python_module>:<FastAPI_instance> --host <host> --port <port>
# Example: "app.main:app" means "import app from main.py that is inside a folder named app".
# Port 8000 is the internal port where the API listens.
# Host 0.0.0.0 allows connections from outside the container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]