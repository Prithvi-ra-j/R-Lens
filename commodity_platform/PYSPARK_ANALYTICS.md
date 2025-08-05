# 🚀 PySpark Analytics Module

This document describes the advanced analytics capabilities powered by Apache Spark for processing large-scale commodity data.

## 🎯 Overview

The PySpark analytics module processes **1M+ records** of commodity price data to calculate comprehensive KPIs including:

- **Daily Analytics**: Average prices, volatility, daily returns
- **Weekly Analytics**: Weekly highs/lows, weekly returns, volatility
- **Monthly Analytics**: Monthly aggregations and trends
- **Overall Statistics**: All-time highs/lows, overall volatility
- **Price Momentum**: Price change patterns and win rates

## 📊 Features

### ✅ **Large-Scale Data Processing**
- Processes **1M+** records efficiently using PySpark
- Generates realistic commodity price data with trends and seasonality
- Supports both sample data generation and real database integration

### ✅ **Comprehensive KPIs**
- **5 KPI Categories**: Daily, Weekly, Monthly, Overall, Momentum
- **Advanced Metrics**: Volatility, returns, price ranges, win rates
- **Window Functions**: Price momentum and trend analysis

### ✅ **Optimized Storage**
- **Parquet Format**: Columnar storage for fast analytics
- **CSV Export**: Human-readable format for external tools
- **Partitioned by Commodity**: Optimized for query performance

### ✅ **Automated Scheduling**
- **Daily ETL Jobs**: Scheduled at 2 AM UTC using APScheduler
- **Weekly Jobs**: Comprehensive processing with 5M records
- **Job Persistence**: SQLite-based job store for reliability

### ✅ **API Integration**
- **RESTful Endpoints**: Fast analytics data access
- **Background Processing**: Non-blocking ETL job execution
- **Status Monitoring**: Real-time job status and history

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │────│  PySpark ETL     │────│  Output Storage │
│                 │    │                  │    │                 │
│ • SQLite DB     │    │ • Data Loading   │    │ • Parquet Files │
│ • Sample Data   │    │ • KPI Calculation│    │ • CSV Files     │
│ • Yahoo Finance │    │ • Transformations│    │ • JSON Metadata │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   APScheduler    │
                       │                  │
                       │ • Daily Jobs     │
                       │ • Weekly Jobs    │
                       │ • Job Monitoring │
                       └──────────────────┘
```

## 🛠️ Usage

### **1. Standalone ETL Execution**

```bash
# Run with default settings (1M records)
python3 run_pyspark_etl.py

# Run with custom record count
python3 run_pyspark_etl.py --records 5000000

# Use database data instead of sample
python3 run_pyspark_etl.py --use-database
```

### **2. API-Triggered ETL**

```bash
# Start ETL job via API
curl -X POST "http://localhost:8000/analytics/etl/run?num_records=1000000"

# Check ETL status
curl "http://localhost:8000/analytics/etl/status"
```

### **3. Scheduled ETL (Automatic)**

The scheduler runs automatically when the FastAPI app starts:
- **Daily Job**: 2 AM UTC, 1M records
- **Weekly Job**: Sunday 1 AM UTC, 5M records

### **4. Analytics Data Access**

```bash
# Get available commodities
curl "http://localhost:8000/analytics/commodities"

# Get overall KPIs for Gold
curl "http://localhost:8000/analytics/GOLD?kpi_type=overall"

# Get daily analytics for Silver
curl "http://localhost:8000/analytics/SILVER?kpi_type=daily"

# Get comprehensive summary
curl "http://localhost:8000/analytics/GOLD/summary"
```

## 📊 KPI Categories

### **1. Overall Statistics**
```json
{
  "commodity": "GOLD",
  "overall_avg_price": 2025.67,
  "all_time_high": 2150.23,
  "all_time_low": 1875.45,
  "total_observations": 166667,
  "overall_volatility": 32.15,
  "price_range": 274.78,
  "volatility_percent": 1.59
}
```

### **2. Daily Analytics**
```json
{
  "commodity": "GOLD",
  "date": "2024-01-15",
  "avg_daily_price": 2023.45,
  "daily_low": 2010.22,
  "daily_high": 2035.67,
  "daily_observations": 24,
  "daily_volatility": 8.45,
  "daily_range": 25.45,
  "daily_return": 0.67
}
```

### **3. Weekly Analytics**
```json
{
  "commodity": "GOLD",
  "year": 2024,
  "week": 3,
  "avg_weekly_price": 2025.33,
  "weekly_low": 1995.22,
  "weekly_high": 2055.67,
  "weekly_observations": 168,
  "weekly_volatility": 15.23,
  "weekly_range": 60.45,
  "weekly_return": 1.23
}
```

### **4. Monthly Analytics**
```json
{
  "commodity": "GOLD",
  "year": 2024,
  "month": 1,
  "avg_monthly_price": 2020.15,
  "monthly_low": 1980.45,
  "monthly_high": 2070.22,
  "monthly_observations": 744,
  "monthly_volatility": 22.67,
  "monthly_range": 89.77,
  "monthly_return": 2.15
}
```

### **5. Momentum Analytics**
```json
{
  "commodity": "GOLD",
  "avg_price_change": 0.23,
  "avg_price_change_pct": 0.012,
  "momentum_volatility": 1.45,
  "positive_moves": 83421,
  "negative_moves": 83245,
  "win_rate": 50.05
}
```

## 🗂️ Output Files

The ETL process generates the following files in `/data/analytics/`:

```
data/analytics/
├── overall_kpis/              # Parquet partitioned by commodity
│   ├── commodity=GOLD/
│   ├── commodity=SILVER/
│   └── ...
├── daily_kpis/               # Daily analytics
├── weekly_kpis/              # Weekly analytics  
├── monthly_kpis/             # Monthly analytics
├── momentum_kpis/            # Momentum analytics
├── overall_kpis_csv/         # CSV exports
├── daily_kpis_csv/
├── weekly_kpis_csv/
├── monthly_kpis_csv/
├── momentum_kpis_csv/
├── analytics_summary.json    # ETL metadata
└── job_log.json             # Job execution history
```

## ⚡ Performance

### **Processing Capabilities**
- **1M Records**: ~2-3 minutes
- **5M Records**: ~8-12 minutes  
- **Memory Usage**: ~2GB driver, ~2GB executor
- **Output Size**: ~50-100MB for 1M records

### **Optimizations**
- **Adaptive Query Execution**: Automatic optimization
- **Partition Coalescing**: Reduces small files
- **Arrow Integration**: Fast Pandas interop
- **Columnar Storage**: Parquet format for analytics

## 🔧 Configuration

### **Spark Configuration**
```python
spark = SparkSession.builder \
    .appName("CommodityAnalytics") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()
```

### **Scheduler Configuration**
```python
# Daily job at 2 AM UTC
scheduler.add_job(
    func=run_daily_etl_job,
    trigger=CronTrigger(hour=2, minute=0),
    id='daily_etl'
)

# Weekly job on Sunday at 1 AM UTC
scheduler.add_job(
    func=run_comprehensive_etl,
    trigger=CronTrigger(day_of_week=6, hour=1),
    id='weekly_etl'
)
```

## 🎯 Use Cases

### **1. Risk Management**
- Monitor volatility across different timeframes
- Calculate Value at Risk (VaR) metrics
- Track price momentum and trends

### **2. Trading Analytics**
- Analyze daily/weekly return patterns
- Calculate win rates and success metrics
- Identify optimal trading windows

### **3. Portfolio Analysis**
- Compare volatility across commodities
- Analyze correlation patterns
- Monitor portfolio exposure

### **4. Business Intelligence**
- Generate executive dashboards
- Create automated reports
- Monitor market conditions

## 🚀 Scaling

### **For Larger Datasets**
```python
# Increase cluster resources
.config("spark.driver.memory", "4g") \
.config("spark.executor.memory", "4g") \
.config("spark.executor.instances", "4") \
.config("spark.executor.cores", "2")

# Optimize partitioning
df.repartition(col("commodity"), col("year"), col("month"))

# Use broadcast for small lookups
spark.sparkContext.broadcast(small_lookup_table)
```

### **For Production**
```python
# Enable dynamic allocation
.config("spark.dynamicAllocation.enabled", "true") \
.config("spark.dynamicAllocation.minExecutors", "1") \
.config("spark.dynamicAllocation.maxExecutors", "10")

# Configure checkpointing
spark.sparkContext.setCheckpointDir("/tmp/checkpoints")
```

## 📈 Monitoring

### **Job Status Monitoring**
```bash
# Get scheduler status
GET /analytics/etl/status

# View job history
GET /analytics/etl/status
```

### **Performance Metrics**
- **Record Processing Rate**: Records/second
- **Job Duration**: Total execution time
- **Memory Usage**: Driver and executor memory
- **File Output Size**: Generated data volume

## 🔮 Future Enhancements

### **Planned Features**
- [ ] **Stream Processing**: Real-time analytics with Spark Streaming
- [ ] **Advanced ML**: MLlib integration for predictive analytics
- [ ] **Data Quality**: Automated data validation and cleansing
- [ ] **Delta Lake**: ACID transactions and time travel
- [ ] **Kubernetes**: Cloud-native deployment

### **Integration Opportunities**
- [ ] **Apache Airflow**: Advanced workflow orchestration
- [ ] **Kafka**: Real-time data ingestion
- [ ] **ElasticSearch**: Advanced search and analytics
- [ ] **Grafana**: Real-time monitoring dashboards

---

**📚 For more information, see the main [README.md](README.md) and API documentation at `/docs`**