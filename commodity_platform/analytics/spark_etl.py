#!/usr/bin/env python3
"""
PySpark ETL Module for Commodity Price Analytics
Processes large historical commodity price data and calculates KPIs.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import findspark
import pandas as pd

# Initialize Spark
try:
    findspark.init()
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import (
        col, avg, min, max, count, stddev, date_format,
        year, month, dayofmonth, weekofyear, lag, abs as spark_abs,
        when, isnan, isnull, first, last, sum as spark_sum
    )
    from pyspark.sql.window import Window
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
except ImportError as e:
    print(f"Error importing PySpark: {e}")
    print("Please install PySpark: pip install pyspark")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CommoditySparkETL:
    def __init__(self, app_name: str = "CommodityAnalytics"):
        """Initialize Spark session and configure analytics"""
        self.app_name = app_name
        self.spark = None
        self.output_path = "/workspace/commodity_platform/data/analytics"
        self.db_path = "/workspace/commodity_platform/data.db"
        
        # Ensure output directory exists
        os.makedirs(self.output_path, exist_ok=True)
        
        self._init_spark()
    
    def _init_spark(self):
        """Initialize Spark session with optimized configuration"""
        try:
            self.spark = SparkSession.builder \
                .appName(self.app_name) \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \
                .config("spark.driver.memory", "2g") \
                .config("spark.executor.memory", "2g") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                .getOrCreate()
            
            # Set log level to reduce verbose output
            self.spark.sparkContext.setLogLevel("WARN")
            logger.info(f"Spark session initialized: {self.spark.version}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Spark session: {e}")
            raise
    
    def load_sqlite_data(self) -> Optional[object]:
        """Load commodity price data from SQLite database"""
        try:
            logger.info("Loading data from SQLite database...")
            
            # Read from SQLite using pandas first, then convert to Spark DataFrame
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            
            # Get all price data
            query = """
            SELECT commodity, price, timestamp, source
            FROM prices 
            ORDER BY commodity, timestamp
            """
            
            pandas_df = pd.read_sql_query(query, conn)
            conn.close()
            
            if pandas_df.empty:
                logger.warning("No data found in SQLite database")
                return None
            
            logger.info(f"Loaded {len(pandas_df)} records from SQLite")
            
            # Convert pandas DataFrame to Spark DataFrame
            spark_df = self.spark.createDataFrame(pandas_df)
            
            # Convert timestamp string to proper timestamp
            spark_df = spark_df.withColumn(
                "timestamp", 
                col("timestamp").cast(TimestampType())
            )
            
            # Add date column for easier grouping
            spark_df = spark_df.withColumn("date", date_format(col("timestamp"), "yyyy-MM-dd"))
            
            return spark_df
            
        except Exception as e:
            logger.error(f"Error loading SQLite data: {e}")
            return None
    
    def generate_large_sample_data(self, num_records: int = 1000000) -> object:
        """Generate large sample dataset for testing (1M+ rows)"""
        logger.info(f"Generating {num_records:,} sample records...")
        
        try:
            import numpy as np
            from datetime import datetime, timedelta
            
            # Define commodities and their base prices
            commodities = {
                "GOLD": {"base_price": 2000.0, "volatility": 0.02},
                "SILVER": {"base_price": 25.0, "volatility": 0.03},
                "OIL": {"base_price": 75.0, "volatility": 0.04},
                "GAS": {"base_price": 3.5, "volatility": 0.05},
                "COPPER": {"base_price": 8.5, "volatility": 0.03},
                "PLATINUM": {"base_price": 950.0, "volatility": 0.025}
            }
            
            # Generate time series data
            start_date = datetime.now() - timedelta(days=365 * 3)  # 3 years of data
            
            data = []
            records_per_commodity = num_records // len(commodities)
            
            for commodity, params in commodities.items():
                base_price = params["base_price"]
                volatility = params["volatility"]
                
                # Generate timestamps (hourly data)
                timestamps = [
                    start_date + timedelta(hours=i) 
                    for i in range(records_per_commodity)
                ]
                
                # Generate prices with realistic patterns
                prices = []
                current_price = base_price
                
                for i, ts in enumerate(timestamps):
                    # Add trend (slight upward bias)
                    trend = 0.0001 * np.sin(i / 1000)
                    
                    # Add seasonality (daily and weekly patterns)
                    daily_pattern = 0.001 * np.sin(2 * np.pi * ts.hour / 24)
                    weekly_pattern = 0.002 * np.sin(2 * np.pi * ts.weekday() / 7)
                    
                    # Add random walk
                    random_change = np.random.normal(0, volatility)
                    
                    # Calculate new price
                    price_change = trend + daily_pattern + weekly_pattern + random_change
                    current_price *= (1 + price_change)
                    
                    # Ensure price doesn't go negative
                    current_price = max(current_price, base_price * 0.3)
                    
                    prices.append(current_price)
                
                # Create records
                for ts, price in zip(timestamps, prices):
                    data.append({
                        "commodity": commodity,
                        "price": round(price, 2),
                        "timestamp": ts,
                        "source": "Generated",
                        "date": ts.strftime("%Y-%m-%d")
                    })
            
            # Create Spark DataFrame
            schema = StructType([
                StructField("commodity", StringType(), True),
                StructField("price", DoubleType(), True),
                StructField("timestamp", TimestampType(), True),
                StructField("source", StringType(), True),
                StructField("date", StringType(), True)
            ])
            
            spark_df = self.spark.createDataFrame(data, schema)
            logger.info(f"Generated {spark_df.count():,} records across {len(commodities)} commodities")
            
            return spark_df
            
        except Exception as e:
            logger.error(f"Error generating sample data: {e}")
            raise
    
    def calculate_kpis(self, df: object) -> Dict[str, object]:
        """Calculate comprehensive KPIs for each commodity"""
        logger.info("Calculating KPIs...")
        
        try:
            kpis = {}
            
            # 1. Daily KPIs
            logger.info("Calculating daily KPIs...")
            daily_kpis = df.groupBy("commodity", "date") \
                .agg(
                    avg("price").alias("avg_daily_price"),
                    min("price").alias("daily_low"),
                    max("price").alias("daily_high"),
                    count("price").alias("daily_observations"),
                    stddev("price").alias("daily_volatility"),
                    first("price").alias("open_price"),
                    last("price").alias("close_price")
                ) \
                .withColumn("daily_range", col("daily_high") - col("daily_low")) \
                .withColumn("daily_return", 
                           (col("close_price") - col("open_price")) / col("open_price") * 100)
            
            kpis["daily"] = daily_kpis
            
            # 2. Weekly KPIs
            logger.info("Calculating weekly KPIs...")
            df_with_week = df.withColumn("year", year("timestamp")) \
                            .withColumn("week", weekofyear("timestamp"))
            
            weekly_kpis = df_with_week.groupBy("commodity", "year", "week") \
                .agg(
                    avg("price").alias("avg_weekly_price"),
                    min("price").alias("weekly_low"),
                    max("price").alias("weekly_high"),
                    count("price").alias("weekly_observations"),
                    stddev("price").alias("weekly_volatility"),
                    first("price").alias("week_open"),
                    last("price").alias("week_close")
                ) \
                .withColumn("weekly_range", col("weekly_high") - col("weekly_low")) \
                .withColumn("weekly_return", 
                           (col("week_close") - col("week_open")) / col("week_open") * 100)
            
            kpis["weekly"] = weekly_kpis
            
            # 3. Monthly KPIs
            logger.info("Calculating monthly KPIs...")
            df_with_month = df.withColumn("year", year("timestamp")) \
                             .withColumn("month", month("timestamp"))
            
            monthly_kpis = df_with_month.groupBy("commodity", "year", "month") \
                .agg(
                    avg("price").alias("avg_monthly_price"),
                    min("price").alias("monthly_low"),
                    max("price").alias("monthly_high"),
                    count("price").alias("monthly_observations"),
                    stddev("price").alias("monthly_volatility"),
                    first("price").alias("month_open"),
                    last("price").alias("month_close")
                ) \
                .withColumn("monthly_range", col("monthly_high") - col("monthly_low")) \
                .withColumn("monthly_return", 
                           (col("month_close") - col("month_open")) / col("month_open") * 100)
            
            kpis["monthly"] = monthly_kpis
            
            # 4. Overall commodity statistics
            logger.info("Calculating overall statistics...")
            overall_stats = df.groupBy("commodity") \
                .agg(
                    avg("price").alias("overall_avg_price"),
                    min("price").alias("all_time_low"),
                    max("price").alias("all_time_high"),
                    count("price").alias("total_observations"),
                    stddev("price").alias("overall_volatility")
                ) \
                .withColumn("price_range", col("all_time_high") - col("all_time_low")) \
                .withColumn("volatility_percent", col("overall_volatility") / col("overall_avg_price") * 100)
            
            kpis["overall"] = overall_stats
            
            # 5. Price momentum (using window functions)
            logger.info("Calculating price momentum...")
            window_spec = Window.partitionBy("commodity").orderBy("timestamp")
            
            momentum_df = df.withColumn(
                "prev_price", lag("price", 1).over(window_spec)
            ).withColumn(
                "price_change", col("price") - col("prev_price")
            ).withColumn(
                "price_change_pct", 
                when(col("prev_price").isNotNull(), 
                     (col("price_change") / col("prev_price")) * 100).otherwise(0)
            )
            
            momentum_stats = momentum_df.groupBy("commodity") \
                .agg(
                    avg("price_change").alias("avg_price_change"),
                    avg("price_change_pct").alias("avg_price_change_pct"),
                    stddev("price_change_pct").alias("momentum_volatility"),
                    count(when(col("price_change") > 0, 1)).alias("positive_moves"),
                    count(when(col("price_change") < 0, 1)).alias("negative_moves")
                ) \
                .withColumn("win_rate", 
                           col("positive_moves") / (col("positive_moves") + col("negative_moves")) * 100)
            
            kpis["momentum"] = momentum_stats
            
            logger.info("KPI calculation completed")
            return kpis
            
        except Exception as e:
            logger.error(f"Error calculating KPIs: {e}")
            raise
    
    def save_analytics_data(self, kpis: Dict[str, object], commodity: str = None):
        """Save KPI data to Parquet and CSV files"""
        logger.info("Saving analytics data...")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for kpi_type, df in kpis.items():
                if commodity:
                    # Filter for specific commodity
                    df = df.filter(col("commodity") == commodity.upper())
                
                if df.count() == 0:
                    logger.warning(f"No data to save for {kpi_type}")
                    continue
                
                # Define output paths
                parquet_path = f"{self.output_path}/{kpi_type}_kpis"
                csv_path = f"{self.output_path}/{kpi_type}_kpis_csv"
                
                # Save as Parquet (partitioned by commodity)
                logger.info(f"Saving {kpi_type} KPIs to Parquet...")
                df.write \
                    .mode("overwrite") \
                    .partitionBy("commodity") \
                    .parquet(parquet_path)
                
                # Save as CSV (coalesced to avoid too many small files)
                logger.info(f"Saving {kpi_type} KPIs to CSV...")
                df.coalesce(1) \
                    .write \
                    .mode("overwrite") \
                    .option("header", "true") \
                    .csv(csv_path)
                
                logger.info(f"Saved {kpi_type} KPIs - Records: {df.count()}")
            
            # Create summary file
            self._create_summary_file(kpis, timestamp)
            
        except Exception as e:
            logger.error(f"Error saving analytics data: {e}")
            raise
    
    def _create_summary_file(self, kpis: Dict[str, object], timestamp: str):
        """Create a summary JSON file with metadata"""
        try:
            summary = {
                "generated_at": timestamp,
                "spark_version": self.spark.version,
                "kpi_types": list(kpis.keys()),
                "record_counts": {},
                "commodities": []
            }
            
            # Get record counts and commodities
            for kpi_type, df in kpis.items():
                summary["record_counts"][kpi_type] = df.count()
                if kpi_type == "overall":
                    commodities = [row["commodity"] for row in df.select("commodity").collect()]
                    summary["commodities"] = commodities
            
            # Save summary
            import json
            summary_path = f"{self.output_path}/analytics_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Summary saved to {summary_path}")
            
        except Exception as e:
            logger.warning(f"Could not create summary file: {e}")
    
    def load_analytics_data(self, commodity: str, kpi_type: str = "overall") -> Optional[Dict]:
        """Load analytics data for API endpoint"""
        try:
            parquet_path = f"{self.output_path}/{kpi_type}_kpis"
            
            if not os.path.exists(parquet_path):
                logger.warning(f"Analytics data not found at {parquet_path}")
                return None
            
            # Read Parquet file
            df = self.spark.read.parquet(parquet_path)
            
            # Filter for specific commodity
            commodity_df = df.filter(col("commodity") == commodity.upper())
            
            if commodity_df.count() == 0:
                logger.warning(f"No analytics data found for {commodity}")
                return None
            
            # Convert to dict
            result = commodity_df.collect()
            data = [row.asDict() for row in result]
            
            return {
                "commodity": commodity.upper(),
                "kpi_type": kpi_type,
                "data": data,
                "record_count": len(data),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error loading analytics data: {e}")
            return None
    
    def run_etl_pipeline(self, use_sample_data: bool = True, num_records: int = 1000000):
        """Run the complete ETL pipeline"""
        logger.info("Starting PySpark ETL pipeline...")
        
        try:
            # Load data
            if use_sample_data:
                df = self.generate_large_sample_data(num_records)
            else:
                df = self.load_sqlite_data()
            
            if df is None:
                logger.error("No data available for processing")
                return False
            
            logger.info(f"Processing {df.count():,} records")
            
            # Calculate KPIs
            kpis = self.calculate_kpis(df)
            
            # Save results
            self.save_analytics_data(kpis)
            
            logger.info("ETL pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"ETL pipeline failed: {e}")
            return False
        
        finally:
            # Clean up Spark session
            if self.spark:
                self.spark.stop()
    
    def close(self):
        """Close Spark session"""
        if self.spark:
            self.spark.stop()
            logger.info("Spark session closed")

def main():
    """Main function for standalone execution"""
    print("🚀 PySpark Commodity Analytics ETL")
    print("=" * 50)
    
    # Initialize ETL
    etl = CommoditySparkETL()
    
    try:
        # Run ETL pipeline
        success = etl.run_etl_pipeline(use_sample_data=True, num_records=1000000)
        
        if success:
            print("✅ ETL pipeline completed successfully!")
            print(f"📊 Analytics data saved to: {etl.output_path}")
        else:
            print("❌ ETL pipeline failed!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        etl.close()

if __name__ == "__main__":
    main()