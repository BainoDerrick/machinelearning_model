FROM python:3.9-slim

WORKDIR /app

# Install PyTorch and PyG dependencies
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    gcc \
    python3-dev

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]