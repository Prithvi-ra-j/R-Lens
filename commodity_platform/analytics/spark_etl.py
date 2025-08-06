import os
import logging
from datetime import datetime, timedelta
import random

from pyspark.sql import SparkSession
from pyspark.sql import DataFrame, Row
from pyspark.sql.functions import col, year, month, avg
from pyspark.sql.types import StructType, StructField, StringType, FloatType, TimestampType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("analytics.spark_etl")


class CommoditySparkETL:
    """Class to handle PySpark ETL for commodity analytics."""

    def __init__(self, app_name="CommodityETL"):
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .getOrCreate()
        logger.info(f"Spark session initialized: {self.spark.version}")
        self.output_path = os.path.abspath("output")

    def generate_sample_data(self, num_records: int) -> DataFrame:
        """Generates synthetic commodity trading data as a Spark DataFrame"""
        try:
            commodities = ['Gold', 'Silver', 'Crude Oil', 'Natural Gas', 'Wheat', 'Corn']
            countries = ['USA', 'India', 'China', 'Germany', 'Brazil', 'South Africa']
            now = datetime.now()

            rows = []
            for _ in range(num_records):
                commodity = random.choice(commodities)
                country = random.choice(countries)
                date = now - timedelta(days=random.randint(0, 365))
                open_price = round(random.uniform(100, 1000), 2)
                close_price = round(open_price + random.uniform(-50, 50), 2)
                low_price = round(min(open_price, close_price) - random.uniform(0, 10), 2)
                high_price = round(max(open_price, close_price) + random.uniform(0, 10), 2)
                volume = round(random.uniform(1000, 100000), 2)

                rows.append(Row(
                    commodity=commodity,
                    country=country,
                    date=date,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume
                ))

            schema = StructType([
                StructField("commodity", StringType(), True),
                StructField("country", StringType(), True),
                StructField("date", TimestampType(), True),
                StructField("open", FloatType(), True),
                StructField("high", FloatType(), True),
                StructField("low", FloatType(), True),
                StructField("close", FloatType(), True),
                StructField("volume", FloatType(), True)
            ])

            df = self.spark.createDataFrame(rows, schema=schema)
            return df

        except Exception as e:
            logger.error(f"Error generating sample data: {e}")
            raise

    def transform_data(self, df: DataFrame) -> DataFrame:
        """Transforms the input DataFrame into aggregated commodity analytics"""
        logger.info("Transforming data...")

        df_transformed = df.withColumn("year", year(col("date"))) \
            .withColumn("month", month(col("date")))

        df_grouped = df_transformed.groupBy("commodity", "country", "year", "month") \
            .agg(
                avg("open").alias("avg_open"),
                avg("close").alias("avg_close"),
                avg("high").alias("avg_high"),
                avg("low").alias("avg_low"),
                avg("volume").alias("avg_volume")
            ).orderBy("commodity", "country", "year", "month")

        return df_grouped

    def save_output(self, df: DataFrame):
        """Saves the transformed DataFrame to disk as JSON"""
        logger.info(f"Saving analytics output to {self.output_path}")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        df.coalesce(1).write.mode("overwrite").json(self.output_path)

    def run_etl_pipeline(self, use_sample_data=True, num_records=1000000) -> bool:
        """Runs the complete ETL pipeline"""
        try:
            if use_sample_data:
                logger.info("Starting PySpark ETL pipeline...")
                logger.info(f"Generating {num_records:,} sample records...")
                df = self.generate_sample_data(num_records)
            else:
                raise NotImplementedError("Database source not implemented yet.")

            logger.info("Transforming sample data...")
            df_transformed = self.transform_data(df)

            logger.info("Saving transformed data...")
            self.save_output(df_transformed)

            return True

        except Exception as e:
            logger.error(f"ETL pipeline failed: {e}")
            return False

    def close(self):
        """Stops the Spark session"""
        self.spark.stop()
        logger.info("Spark session closed")
