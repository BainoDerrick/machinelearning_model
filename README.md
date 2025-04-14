# East Coast Fever Risk Prediction System

🌾 GNN-based risk prediction with knowledge distillation

## Setup
```bash
git clone https://your-repo-url.git
cd your_model_project
pip install -r requirements.txt
```

## Run Locally
```bash
streamlit run app/app.py
```

## Build Docker Image
```bash
docker build -t farm-risk-app .
docker run -p 8501:8501 farm-risk-app
```

## Deployment
1. Push to GitHub
2. Deploy on [Crane Cloud](https://cranecloud.io) using the Dockerfile