# 🚀 Quick Setup Guide

## 📋 Prerequisites

Ensure you have Python 3.8+ installed on your system.

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Sample Data & Train PyTorch Models

```bash
# Generate sample historical data
cd data/historical
python3 generate_sample_data.py
cd ../..

# Train PyTorch LSTM models (recommended)
python3 train_pytorch_models.py
```

This creates sample historical data and trains PyTorch models for predictions.

### 3. Start the Platform

**Option A: Using the runner script (Recommended)**
```bash
python3 run.py
```

**Option B: Start services manually**

Terminal 1 (API):
```bash
cd api
python3 main.py
```

Terminal 2 (Dashboard):
```bash
cd dashboard
streamlit run app.py
```

## 🌐 Access Points

- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501

## 🧪 Test the Platform

1. Visit http://localhost:8000/health to check API status
2. Open http://localhost:8501 for the interactive dashboard
3. Try creating alert rules and viewing price charts

## 📊 Sample Features

- View real-time commodity prices
- Create price alerts (e.g., Gold above $2100)
- Generate AI price predictions using PyTorch LSTM models
- Monitor alert history
- Train custom PyTorch models with your data

## 🔧 Troubleshooting

- If dependencies fail to install, try: `pip install --upgrade pip`
- For PyTorch GPU support: Visit https://pytorch.org/get-started/locally/
- If PyTorch training is slow, consider using GPU: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- If API can't connect, check firewall settings for ports 8000/8501