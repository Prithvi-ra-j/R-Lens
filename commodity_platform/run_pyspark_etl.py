#!/usr/bin/env python3
"""
Standalone PySpark ETL Runner
Run this script to execute the commodity analytics ETL pipeline
"""

import sys
import os
import argparse
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analytics.spark_etl import CommoditySparkETL

def main():
    """Main function to run PySpark ETL"""
    parser = argparse.ArgumentParser(description='Run PySpark ETL for Commodity Analytics')
    parser.add_argument('--records', type=int, default=1000000, 
                       help='Number of records to generate (default: 1,000,000)')
    parser.add_argument('--use-sample', action='store_true', default=True,
                       help='Use generated sample data (default: True)')
    parser.add_argument('--use-database', action='store_true', 
                       help='Use data from SQLite database instead of sample data')
    
    args = parser.parse_args()
    
    print("🚀 Commodity Platform - PySpark ETL Runner")
    print("=" * 60)
    print(f"📊 Records to process: {args.records:,}")
    print(f"📁 Data source: {'Database' if args.use_database else 'Generated Sample'}")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Initialize ETL
        etl = CommoditySparkETL(app_name="StandaloneETL")
        
        # Run ETL pipeline
        success = etl.run_etl_pipeline(
            use_sample_data=not args.use_database,
            num_records=args.records
        )
        
        if success:
            print("\n✅ ETL Pipeline completed successfully!")
            print(f"📊 Analytics data saved to: {etl.output_path}")
            
            # Show output files
            if os.path.exists(etl.output_path):
                print("\n📁 Generated Files:")
                for item in os.listdir(etl.output_path):
                    item_path = os.path.join(etl.output_path, item)
                    if os.path.isdir(item_path):
                        size = sum(
                            os.path.getsize(os.path.join(item_path, f))
                            for f in os.listdir(item_path)
                            if os.path.isfile(os.path.join(item_path, f))
                        )
                        print(f"   📂 {item}: {size / 1024 / 1024:.1f} MB")
                    elif item.endswith('.json'):
                        size = os.path.getsize(item_path)
                        print(f"   📄 {item}: {size / 1024:.1f} KB")
            
            print("\n🎯 Next Steps:")
            print("   1. Start the API: cd api && python main.py")
            print("   2. View analytics: http://localhost:8000/analytics/commodities")
            print("   3. Start dashboard: cd dashboard && streamlit run app.py")
            
        else:
            print("\n❌ ETL Pipeline failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error running ETL: {e}")
        sys.exit(1)
    
    finally:
        # Cleanup
        try:
            etl.close()
        except:
            pass
    
    end_time = datetime.now()
    print(f"\n🏁 Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()