#!/usr/bin/env python3
"""
PyTorch Model Training Script for Commodity Platform
This script trains LSTM models using PyTorch for all supported commodities.
"""

import sys
import os
sys.path.append('commodity_platform')

from api.db import DatabaseHandler
from api.ml_model import CommodityPredictor
import torch

def check_pytorch():
    """Check PyTorch installation and GPU availability"""
    print("🔧 PyTorch Setup Check")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("Using CPU for training")
    print()

def train_all_models():
    """Train PyTorch models for all commodities"""
    print("🚀 PyTorch LSTM Model Training")
    print("=" * 50)
    
    # Initialize components
    db_handler = DatabaseHandler()
    predictor = CommodityPredictor()
    
    commodities = ['gold', 'silver', 'oil', 'gas']
    results = {}
    
    for commodity in commodities:
        print(f"\n📈 Training model for {commodity.upper()}")
        print("-" * 30)
        
        try:
            # Get historical data
            df = db_handler.get_historical_data(commodity, days=200)
            
            if len(df) < 70:
                print(f"❌ Insufficient data for {commodity}: {len(df)} records")
                print("   Need at least 70 records for training")
                results[commodity] = {
                    'success': False,
                    'error': 'Insufficient data',
                    'data_points': len(df)
                }
                continue
            
            print(f"✅ Found {len(df)} data points for {commodity}")
            print("🔄 Starting PyTorch training...")
            
            # Train the model
            result = predictor.train_model(commodity, df, epochs=30)
            results[commodity] = result
            
            if result['success']:
                print(f"✅ Training completed successfully!")
                print(f"   - Train RMSE: {result['train_rmse']:.6f}")
                print(f"   - Test RMSE: {result['test_rmse']:.6f}")
                print(f"   - Test MAE: {result['test_mae']:.6f}")
                print(f"   - Model Accuracy: {result['model_accuracy']:.2%}")
                print(f"   - Framework: {result['framework']}")
            else:
                print(f"❌ Training failed: {result['error']}")
                
        except Exception as e:
            print(f"❌ Error training {commodity}: {e}")
            results[commodity] = {
                'success': False,
                'error': str(e),
                'framework': 'PyTorch'
            }
    
    return results

def test_predictions():
    """Test predictions with trained models"""
    print("\n🔮 Testing PyTorch Predictions")
    print("=" * 50)
    
    predictor = CommodityPredictor()
    commodities = ['gold', 'silver', 'oil', 'gas']
    
    for commodity in commodities:
        print(f"\n📊 Testing {commodity.upper()} predictions...")
        
        try:
            predictions = predictor.predict_prices(commodity, days_ahead=3)
            
            if predictions:
                print(f"✅ Predictions generated successfully!")
                print(f"   Framework: {predictions.get('framework', 'Unknown')}")
                print(f"   Confidence: {predictions.get('confidence_score', 0):.1%}")
                print(f"   Accuracy: {predictions.get('model_accuracy', 0):.1%}")
                
                print("   Next 3 days:")
                for pred in predictions['predictions']:
                    print(f"     {pred['date']}: ${pred['predicted_price']:.2f}")
                
                if 'note' in predictions:
                    print(f"   Note: {predictions['note']}")
            else:
                print(f"❌ No predictions available for {commodity}")
                
        except Exception as e:
            print(f"❌ Error testing {commodity}: {e}")

def show_model_info():
    """Show information about saved models"""
    print("\n📁 PyTorch Model Files")
    print("=" * 50)
    
    model_dir = "commodity_platform/models"
    if not os.path.exists(model_dir):
        print("❌ Models directory not found")
        return
    
    commodities = ['gold', 'silver', 'oil', 'gas']
    
    for commodity in commodities:
        model_file = os.path.join(model_dir, f"{commodity}_lstm_pytorch.pth")
        scaler_file = os.path.join(model_dir, f"{commodity}_scaler.npy")
        
        print(f"\n{commodity.upper()}:")
        
        if os.path.exists(model_file):
            size = os.path.getsize(model_file) / 1024
            print(f"  ✅ Model: {model_file} ({size:.1f} KB)")
        else:
            print(f"  ❌ Model: Not found")
        
        if os.path.exists(scaler_file):
            size = os.path.getsize(scaler_file) / 1024
            print(f"  ✅ Scaler: {scaler_file} ({size:.1f} KB)")
        else:
            print(f"  ❌ Scaler: Not found")

def main():
    """Main training function"""
    print("🎯 PyTorch LSTM Model Training for Commodity Platform")
    print("=" * 60)
    
    # Check PyTorch setup
    check_pytorch()
    
    # Check if we have sample data
    db_handler = DatabaseHandler()
    total_records = len(db_handler.get_prices(limit=10000))
    
    if total_records < 100:
        print("⚠️  Warning: Very little sample data found!")
        print("💡 Consider running the sample data generator first:")
        print("   cd data/historical && python3 generate_sample_data.py")
        print()
    else:
        print(f"✅ Found {total_records} total price records in database")
        print()
    
    # Train all models
    results = train_all_models()
    
    # Show results summary
    print("\n📋 Training Results Summary")
    print("=" * 50)
    
    successful = 0
    failed = 0
    
    for commodity, result in results.items():
        status = "✅ Success" if result['success'] else "❌ Failed"
        print(f"{commodity.upper()}: {status}")
        if result['success']:
            successful += 1
        else:
            failed += 1
            print(f"   Error: {result['error']}")
    
    print(f"\nTotal: {successful} successful, {failed} failed")
    
    # Test predictions
    if successful > 0:
        test_predictions()
    
    # Show model files
    show_model_info()
    
    print("\n🎉 PyTorch model training completed!")
    print("💡 You can now use the trained models for predictions in the API and dashboard.")

if __name__ == "__main__":
    main()