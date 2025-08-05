# 🚀 Intelligent Commodity Price Monitoring & Prediction Platform

A comprehensive FastAPI-powered service that fetches real-time commodity prices, manages user-defined alerts, and provides AI-powered price predictions using LSTM neural networks.

## 🎯 Features

- **Real-time Price Monitoring**: Fetches live prices for Gold, Silver, Oil, and Natural Gas
- **Intelligent Alerts**: User-configurable price threshold alerts
- **AI Price Predictions**: LSTM-based machine learning models for price forecasting
- **Big Data Analytics**: PySpark-powered ETL for processing 1M+ records
- **Advanced KPIs**: Daily, weekly, monthly analytics with volatility calculations
- **Interactive Dashboard**: Beautiful Streamlit-powered web interface
- **Data Persistence**: SQLite database + Parquet files for analytics
- **RESTful API**: Complete FastAPI backend with comprehensive endpoints
- **Scheduled ETL**: APScheduler for automated daily analytics processing

## 🏗️ Project Structure

```
commodity_platform/
├── api/                    # FastAPI backend
│   ├── main.py             # API endpoints and FastAPI app
│   ├── prices.py           # Real-time price fetching
│   ├── models.py           # Pydantic data models
│   ├── db.py               # SQLite database handler
│   ├── alert_logic.py      # Alert checking logic
│   └── ml_model.py         # LSTM model training & inference
├── analytics/              # PySpark analytics module
│   ├── spark_etl.py        # PySpark ETL pipeline
│   └── scheduler.py        # APScheduler for automated jobs
├── dashboard/              # Streamlit frontend
│   ├── app.py              # Main dashboard application
│   └── utils.py            # Utility functions for charts & API calls
├── data/
│   ├── historical/         # Historical price data (CSV files)
│   ├── analytics/          # PySpark output (Parquet & CSV files)
│   └── generate_sample_data.py  # Script to generate test data
├── models/                 # Trained LSTM models storage
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── run_pyspark_etl.py      # Standalone PySpark ETL runner
└── data.db                # SQLite database (created automatically)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd commodity_platform
pip install -r requirements.txt
```

### 2. Generate Sample Data & Run PySpark Analytics

```bash
# Generate sample historical data
cd data/historical
python3 generate_sample_data.py
cd ../..

# Run PySpark ETL to generate analytics (1M+ records)
python3 run_pyspark_etl.py
```

This creates sample historical data for testing and initial model training.

### 3. Start the FastAPI Backend

```bash
cd api
python main.py
```

The API will start on `http://localhost:8000`

### 4. Launch the Streamlit Dashboard

Open a new terminal and run:

```bash
cd dashboard
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📊 Dashboard Features

### Overview Page
- Current commodity prices with live updates
- Platform statistics and metrics
- Recent alerts summary
- Auto-refresh capability

### Price Charts
- Interactive price history charts
- Configurable time ranges
- AI predictions overlay
- Price statistics and metrics

### Alert Management
- Create custom price threshold alerts
- View active alert rules
- Alert history with timeline visualization
- Real-time alert notifications

### AI Predictions
- LSTM-powered price forecasting
- Configurable prediction horizons (1-30 days)
- Model confidence scores
- Historical vs. predicted price comparisons

### Settings & Configuration
- API connection testing
- Database statistics
- ML model training interface
- System configuration options

## 🔌 API Endpoints

### Price Endpoints
- `GET /prices/current` - Get current prices for all commodities
- `GET /prices/{commodity}` - Get current price for specific commodity
- `GET /prices/history/{commodity}` - Get price history

### Alert Endpoints
- `POST /alerts/rules` - Create new alert rule
- `GET /alerts/rules` - Get all alert rules
- `GET /alerts` - Get recent alerts
- `GET /alerts/summary` - Get alert summary

### Prediction Endpoints
- `POST /predictions/{commodity}` - Get price predictions
- `POST /models/train/{commodity}` - Train ML model
- `POST /models/retrain-all` - Retrain all models

### Analytics Endpoints (NEW)
- `GET /analytics/{commodity}` - Get PySpark analytics KPIs
- `GET /analytics/{commodity}/summary` - Get comprehensive analytics summary
- `POST /analytics/etl/run` - Trigger PySpark ETL job manually
- `GET /analytics/etl/status` - Get ETL job status and history
- `GET /analytics/commodities` - Get available commodities with analytics

### System Endpoints
- `GET /health` - Health check
- `GET /stats` - Platform statistics
- `GET /` - API information

## 🤖 Machine Learning (PyTorch)

The platform uses PyTorch-based LSTM (Long Short-Term Memory) neural networks for price prediction:

- **Framework**: PyTorch with CUDA support (if available)
- **Model Architecture**: 3-layer LSTM with dropout regularization
- **Features**: Price sequences with 60-day lookback windows
- **Training**: 80/20 train/test split with Adam optimizer
- **Metrics**: RMSE, MAE, and custom accuracy scores
- **Deployment**: Real-time inference with confidence scoring

### Training Models

Models can be trained via:

1. **Dedicated Script**: `python3 train_pytorch_models.py` (Recommended)
2. **API Endpoint**: `POST /models/train/{commodity}`
3. **Dashboard**: Settings → Model Training
4. **Programmatically**: Using the `CommodityPredictor` class

## 💾 Database Schema

### Prices Table
```sql
CREATE TABLE prices (
    id INTEGER PRIMARY KEY,
    commodity TEXT NOT NULL,
    price REAL NOT NULL,
    timestamp TEXT NOT NULL,
    source TEXT DEFAULT 'API'
);
```

### Alert Rules Table
```sql
CREATE TABLE alert_rules (
    id INTEGER PRIMARY KEY,
    commodity TEXT NOT NULL,
    condition TEXT NOT NULL,
    threshold REAL NOT NULL,
    active BOOLEAN DEFAULT 1,
    created_at TEXT NOT NULL
);
```

### Alerts Table
```sql
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY,
    rule_id INTEGER NOT NULL,
    commodity TEXT NOT NULL,
    price REAL NOT NULL,
    condition TEXT NOT NULL,
    threshold REAL NOT NULL,
    triggered_at TEXT NOT NULL,
    message TEXT NOT NULL
);
```

## 🔧 Configuration

### Environment Variables

You can customize the following via environment variables:

- `DATABASE_PATH`: SQLite database file path
- `MODEL_PATH`: Directory for storing ML models
- `API_PORT`: FastAPI server port (default: 8000)
- `UPDATE_INTERVAL`: Price update frequency in seconds (default: 300)

### Data Sources

The platform supports multiple data sources with automatic fallback:

1. **Yahoo Finance** (Primary): Real-time futures data
2. **Metals API** (Backup): For precious metals
3. **Mock Data** (Development): Realistic simulated prices

## 📈 Performance

- **API Response Time**: < 100ms for most endpoints
- **Price Updates**: Every 5 minutes automatically
- **Model Training**: 2-5 minutes per commodity (depending on data size)
- **Prediction Generation**: < 1 second
- **Database Queries**: Optimized with proper indexing

## 🔒 Security Features

- CORS middleware for web security
- Input validation with Pydantic models
- SQL injection prevention with parameterized queries
- Error handling with appropriate HTTP status codes

## 🧪 Testing

### Manual Testing

1. **API Testing**: Visit `http://localhost:8000/docs` for Swagger UI
2. **Price Fetching**: Check `http://localhost:8000/prices/current`
3. **Health Check**: Visit `http://localhost:8000/health`

### Sample Data

Use the provided sample data generator to create test data:

```bash
python data/historical/generate_sample_data.py
```

## 📦 Dependencies

### Core Dependencies
- **FastAPI**: Modern, fast web framework
- **Streamlit**: Interactive web dashboard
- **PyTorch**: Machine learning framework with GPU support
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive charts and visualizations
- **YFinance**: Financial data provider
- **SQLite**: Lightweight database

### Full Requirements
See `requirements.txt` for complete dependency list with versions.

## 🚀 Deployment

### Local Development
Follow the Quick Start guide above.

### Production Deployment

1. **Docker** (Recommended):
```bash
# Build and run with Docker Compose
docker-compose up --build
```

2. **Manual Deployment**:
```bash
# Install dependencies
pip install -r requirements.txt

# Start API with production server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Start dashboard
streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

For questions, issues, or contributions:

1. **Issues**: Open a GitHub issue
2. **Documentation**: Check the code comments and docstrings
3. **API Docs**: Visit `/docs` endpoint when running the API

## 🔮 Future Enhancements

- [ ] Support for more commodities (Coffee, Sugar, Wheat, etc.)
- [ ] Advanced ML models (Transformer, GRU)
- [ ] Real-time WebSocket updates
- [ ] Email/SMS alert notifications
- [ ] Portfolio tracking and analysis
- [ ] Technical indicator calculations
- [ ] Export functionality for data and reports
- [ ] User authentication and multi-tenancy
- [ ] Mobile app companion

---

**Built with ❤️ using Python, FastAPI, Streamlit, and TensorFlow**