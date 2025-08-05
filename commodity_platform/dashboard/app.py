import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from utils import APIClient, ChartGenerator, format_price, format_timestamp, calculate_price_change, get_status_color

# Page configuration
st.set_page_config(
    page_title="Commodity Price Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize API client
@st.cache_resource
def get_api_client():
    return APIClient()

api_client = get_api_client()

# Sidebar navigation
st.sidebar.title("📈 Commodity Monitor")
page = st.sidebar.selectbox(
    "Navigate to:",
    ["Overview", "Price Charts", "Alerts", "Predictions", "Analytics", "Settings"]
)

# Auto-refresh option
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
if auto_refresh:
    time.sleep(30)
    st.rerun()

# Main content based on selected page
if page == "Overview":
    st.title("📊 Commodity Price Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Current Prices")
        
        # Fetch current prices
        current_prices = api_client.get_current_prices()
        
        if current_prices:
            # Create current prices chart
            chart = ChartGenerator.create_current_prices_chart(current_prices)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            # Display price table
            price_data = []
            for commodity, price_info in current_prices.items():
                price_data.append({
                    'Commodity': commodity.upper(),
                    'Price': format_price(price_info['price']),
                    'Source': price_info['source'],
                    'Last Updated': format_timestamp(price_info['timestamp'])
                })
            
            df = pd.DataFrame(price_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("Unable to fetch current prices. API may be unavailable.")
    
    with col2:
        st.subheader("Platform Statistics")
        
        # Get platform stats
        stats = api_client.get_stats()
        
        if stats:
            st.metric("Total Price Records", stats.get('total_price_records', 0))
            st.metric("Active Alert Rules", stats.get('active_alert_rules', 0))
            st.metric("Alerts Today", stats.get('alerts_today', 0))
            
            if stats.get('available_models'):
                st.write("**Available ML Models:**")
                for model in stats['available_models']:
                    st.write(f"• {model.upper()}")
        
        st.subheader("Recent Alerts")
        
        # Get recent alerts
        alerts_data = api_client.get_alerts(limit=5)
        recent_alerts = alerts_data.get('alerts', [])
        
        if recent_alerts:
            for alert in recent_alerts[:3]:  # Show top 3
                with st.container():
                    condition_color = get_status_color(alert['condition'])
                    st.markdown(
                        f"<div style='padding: 8px; border-left: 4px solid {condition_color}; margin: 4px 0;'>"
                        f"<b>{alert['commodity']}</b><br>"
                        f"{alert['message']}<br>"
                        f"<small>{format_timestamp(alert['triggered_at'])}</small>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
        else:
            st.info("No recent alerts")

elif page == "Price Charts":
    st.title("📈 Price Charts & Historical Data")
    
    # Commodity selection
    commodities = ["gold", "silver", "oil", "gas"]
    selected_commodity = st.selectbox("Select Commodity:", commodities)
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        # Chart settings
        st.subheader("Chart Settings")
        days_history = st.slider("Days of History", 1, 30, 7)
        show_predictions = st.checkbox("Show Predictions", value=True)
        prediction_days = st.slider("Prediction Days", 1, 14, 7) if show_predictions else 7
    
    with col1:
        # Get price history
        history_data = api_client.get_price_history(selected_commodity, limit=days_history * 24)
        prices = history_data.get('prices', [])
        
        # Get predictions if enabled
        predictions = None
        if show_predictions:
            pred_data = api_client.get_predictions(selected_commodity, prediction_days)
            predictions = pred_data.get('predictions', [])
        
        # Create and display chart
        if prices:
            chart = ChartGenerator.create_price_chart(prices, selected_commodity, predictions)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            # Price statistics
            df = pd.DataFrame(prices)
            df['price'] = pd.to_numeric(df['price'])
            
            col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
            
            with col_stats1:
                st.metric("Current Price", format_price(df['price'].iloc[0]))
            
            with col_stats2:
                avg_price = df['price'].mean()
                st.metric("Average Price", format_price(avg_price))
            
            with col_stats3:
                max_price = df['price'].max()
                st.metric("Highest Price", format_price(max_price))
            
            with col_stats4:
                min_price = df['price'].min()
                st.metric("Lowest Price", format_price(min_price))
            
            # Show prediction accuracy if available
            if predictions and pred_data.get('model_accuracy'):
                st.info(f"Model Accuracy: {pred_data['model_accuracy']:.1%}")
        else:
            st.warning(f"No price data available for {selected_commodity}")

elif page == "Alerts":
    st.title("🚨 Alert Management")
    
    tab1, tab2 = st.tabs(["Alert Rules", "Alert History"])
    
    with tab1:
        st.subheader("Create New Alert Rule")
        
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            alert_commodity = st.selectbox("Commodity:", ["gold", "silver", "oil", "gas"])
        
        with col2:
            alert_condition = st.selectbox("Condition:", ["above", "below"])
        
        with col3:
            alert_threshold = st.number_input("Threshold ($):", min_value=0.01, value=100.0, step=0.01)
        
        with col4:
            if st.button("Create Alert", type="primary"):
                result = api_client.create_alert_rule(alert_commodity, alert_condition, alert_threshold)
                if result:
                    st.success("Alert rule created successfully!")
                    st.rerun()
                else:
                    st.error("Failed to create alert rule")
        
        st.subheader("Active Alert Rules")
        
        # Get alert rules
        rules_data = api_client.get_alert_rules()
        rules = rules_data.get('rules', [])
        
        if rules:
            for rule in rules:
                with st.container():
                    col1, col2, col3 = st.columns([3, 2, 1])
                    
                    with col1:
                        status_color = get_status_color('active' if rule['active'] else 'inactive')
                        st.markdown(
                            f"<div style='padding: 8px; border-left: 4px solid {status_color};'>"
                            f"<b>{rule['commodity']}</b> price {rule['condition']} ${rule['threshold']:.2f}<br>"
                            f"<small>Created: {format_timestamp(rule['created_at'])}</small>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        status = "🟢 Active" if rule['active'] else "🔴 Inactive"
                        st.write(status)
                    
                    with col3:
                        # Add toggle button here in a real implementation
                        pass
        else:
            st.info("No alert rules configured")
    
    with tab2:
        st.subheader("Alert History")
        
        # Get alerts
        alerts_data = api_client.get_alerts(limit=50)
        alerts = alerts_data.get('alerts', [])
        
        if alerts:
            # Create alerts timeline
            timeline_chart = ChartGenerator.create_alerts_timeline(alerts)
            if timeline_chart:
                st.plotly_chart(timeline_chart, use_container_width=True)
            
            # Display alerts table
            alerts_df = pd.DataFrame(alerts)
            alerts_df['triggered_at'] = pd.to_datetime(alerts_df['triggered_at'])
            alerts_df = alerts_df.sort_values('triggered_at', ascending=False)
            
            # Format for display
            display_df = alerts_df[['commodity', 'condition', 'price', 'threshold', 'triggered_at', 'message']].copy()
            display_df['price'] = display_df['price'].apply(format_price)
            display_df['threshold'] = display_df['threshold'].apply(format_price)
            display_df['triggered_at'] = display_df['triggered_at'].dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No alerts in history")

elif page == "Predictions":
    st.title("🔮 Price Predictions")
    
    st.info("AI-powered commodity price forecasting using LSTM neural networks")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Prediction Settings")
        pred_commodity = st.selectbox("Select Commodity:", ["gold", "silver", "oil", "gas"])
        pred_days = st.slider("Forecast Days:", 1, 30, 7)
        
        if st.button("Generate Prediction", type="primary"):
            with st.spinner("Generating predictions..."):
                predictions = api_client.get_predictions(pred_commodity, pred_days)
                st.session_state['current_predictions'] = predictions
    
    with col1:
        # Display predictions if available
        if 'current_predictions' in st.session_state and st.session_state['current_predictions']:
            pred_data = st.session_state['current_predictions']
            
            st.subheader(f"{pred_data['commodity']} Price Forecast")
            
            # Create prediction chart
            predictions = pred_data['predictions']
            
            # Get recent price history for context
            history_data = api_client.get_price_history(pred_commodity, limit=30)
            historical_prices = history_data.get('prices', [])
            
            if historical_prices:
                chart = ChartGenerator.create_price_chart(historical_prices, pred_commodity, predictions)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
            
            # Display prediction metrics
            col_metric1, col_metric2 = st.columns(2)
            
            with col_metric1:
                confidence = pred_data.get('confidence_score', 0) * 100
                st.metric("Confidence Score", f"{confidence:.1f}%")
            
            with col_metric2:
                accuracy = pred_data.get('model_accuracy', 0) * 100
                st.metric("Model Accuracy", f"{accuracy:.1f}%")
            
            # Prediction table
            st.subheader("Detailed Forecast")
            pred_df = pd.DataFrame(predictions)
            pred_df['predicted_price'] = pred_df['predicted_price'].apply(format_price)
            st.dataframe(pred_df, use_container_width=True)
        
        else:
            st.info("Select a commodity and click 'Generate Prediction' to see forecasts")
            
            # Show model training status
            st.subheader("Model Training")
            
            if st.button("Train Models"):
                with st.spinner("Training models... This may take several minutes."):
                    for commodity in ["gold", "silver", "oil", "gas"]:
                        result = api_client.train_model(commodity)
                        if result:
                            st.success(f"✅ {commodity.upper()} model trained successfully")
                        else:
                            st.error(f"❌ Failed to train {commodity.upper()} model")

elif page == "Analytics":
    st.title("📊 PySpark Analytics Dashboard")
    
    st.info("Advanced analytics powered by PySpark for large-scale data processing")
    
    # Check analytics status
    try:
        stats = api_client.get_stats()
        analytics_status = stats.get('analytics_status', 'unavailable')
        etl_running = stats.get('etl_scheduler_running', False)
    except:
        analytics_status = 'unavailable'
        etl_running = False
    
    # Status indicators
    col1, col2 = st.columns(2)
    
    with col1:
        if analytics_status == 'available':
            st.success("📊 Analytics Data: Available")
        else:
            st.warning("📊 Analytics Data: Not Available")
    
    with col2:
        if etl_running:
            st.success("🕒 ETL Scheduler: Running")
        else:
            st.warning("🕒 ETL Scheduler: Stopped")
    
    # Analytics controls
    st.subheader("🛠️ Analytics Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Run ETL Job", type="primary"):
            with st.spinner("Starting ETL job..."):
                try:
                    response = requests.post(f"{api_client.base_url}/analytics/etl/run?num_records=1000000")
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ ETL job started successfully!")
                        st.json(result)
                    else:
                        st.error("❌ Failed to start ETL job")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    with col2:
        if st.button("📈 View ETL Status"):
            try:
                response = requests.get(f"{api_client.base_url}/analytics/etl/status")
                if response.status_code == 200:
                    status = response.json()
                    st.json(status)
                else:
                    st.error("Failed to get ETL status")
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col3:
        if st.button("📋 Available Commodities"):
            try:
                response = requests.get(f"{api_client.base_url}/analytics/commodities")
                if response.status_code == 200:
                    commodities = response.json()
                    st.json(commodities)
                else:
                    st.error("Failed to get commodities")
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Analytics viewer
    if analytics_status == 'available':
        st.subheader("📊 Analytics Viewer")
        
        # Get available commodities
        try:
            response = requests.get(f"{api_client.base_url}/analytics/commodities")
            if response.status_code == 200:
                commodities_data = response.json()
                available_commodities = commodities_data.get('commodities', [])
                
                if available_commodities:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        selected_commodity = st.selectbox(
                            "Select Commodity:",
                            available_commodities
                        )
                    
                    with col2:
                        kpi_type = st.selectbox(
                            "Select KPI Type:",
                            ["overall", "daily", "weekly", "monthly", "momentum"]
                        )
                    
                    if st.button("📊 Load Analytics"):
                        with st.spinner(f"Loading {kpi_type} analytics for {selected_commodity}..."):
                            try:
                                response = requests.get(
                                    f"{api_client.base_url}/analytics/{selected_commodity}?kpi_type={kpi_type}"
                                )
                                
                                if response.status_code == 200:
                                    analytics_data = response.json()
                                    
                                    st.success(f"✅ Loaded {analytics_data['record_count']} records")
                                    
                                    # Display summary
                                    st.json({
                                        "commodity": analytics_data['commodity'],
                                        "kpi_type": analytics_data['kpi_type'],
                                        "record_count": analytics_data['record_count'],
                                        "source": analytics_data.get('source', 'Unknown')
                                    })
                                    
                                    # Display data
                                    if analytics_data['data']:
                                        df = pd.DataFrame(analytics_data['data'])
                                        st.dataframe(df, use_container_width=True)
                                        
                                        # Show charts for some KPI types
                                        if kpi_type == "daily" and 'avg_daily_price' in df.columns:
                                            st.subheader("📈 Daily Average Price Trend")
                                            chart_df = df.copy()
                                            if 'date' in chart_df.columns:
                                                chart_df['date'] = pd.to_datetime(chart_df['date'])
                                                st.line_chart(chart_df.set_index('date')['avg_daily_price'])
                                        
                                        elif kpi_type == "overall":
                                            st.subheader("📊 Overall Statistics")
                                            if len(df) > 0:
                                                row = df.iloc[0]
                                                col1, col2, col3, col4 = st.columns(4)
                                                
                                                with col1:
                                                    st.metric("Average Price", f"${row.get('overall_avg_price', 0):.2f}")
                                                
                                                with col2:
                                                    st.metric("All-Time High", f"${row.get('all_time_high', 0):.2f}")
                                                
                                                with col3:
                                                    st.metric("All-Time Low", f"${row.get('all_time_low', 0):.2f}")
                                                
                                                with col4:
                                                    st.metric("Volatility", f"{row.get('volatility_percent', 0):.2f}%")
                                    
                                else:
                                    st.error(f"❌ Failed to load analytics: {response.status_code}")
                                    
                            except Exception as e:
                                st.error(f"❌ Error loading analytics: {e}")
                
                else:
                    st.warning("No commodities with analytics data found")
            
        except Exception as e:
            st.error(f"Error getting available commodities: {e}")
    
    else:
        st.warning("Analytics data not available. Please run an ETL job first.")

elif page == "Settings":
    st.title("⚙️ Settings & Configuration")
    
    tab1, tab2, tab3 = st.tabs(["API Configuration", "Data Management", "Model Training"])
    
    with tab1:
        st.subheader("API Settings")
        
        # Test API connection
        if st.button("Test API Connection"):
            try:
                stats = api_client.get_stats()
                if stats:
                    st.success("✅ API connection successful")
                    st.json(stats)
                else:
                    st.error("❌ API connection failed")
            except Exception as e:
                st.error(f"❌ API connection error: {e}")
        
        st.subheader("Price Update Frequency")
        st.info("Prices are automatically updated every 5 minutes in the background")
    
    with tab2:
        st.subheader("Database Statistics")
        
        stats = api_client.get_stats()
        if stats:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Price Records", stats.get('total_price_records', 0))
            
            with col2:
                st.metric("Supported Commodities", len(stats.get('supported_commodities', [])))
            
            with col3:
                st.metric("Available Models", len(stats.get('available_models', [])))
    
    with tab3:
        st.subheader("ML Model Management")
        
        st.warning("Model training requires sufficient historical data and may take several minutes.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Individual Model Training:**")
            train_commodity = st.selectbox("Select commodity to train:", ["gold", "silver", "oil", "gas"])
            
            if st.button(f"Train {train_commodity.upper()} Model"):
                with st.spinner(f"Training {train_commodity} model..."):
                    result = api_client.train_model(train_commodity)
                    if result:
                        st.success(f"✅ {train_commodity.upper()} model trained successfully")
                        st.json(result)
                    else:
                        st.error(f"❌ Failed to train {train_commodity.upper()} model")
        
        with col2:
            st.write("**Batch Model Training:**")
            st.info("This will retrain all models sequentially")
            
            if st.button("Train All Models", type="primary"):
                with st.spinner("Training all models... This will take several minutes."):
                    # Note: In a real implementation, you'd call a batch training endpoint
                    for commodity in ["gold", "silver", "oil", "gas"]:
                        result = api_client.train_model(commodity)
                        if result:
                            st.success(f"✅ {commodity.upper()} model trained")
                        else:
                            st.error(f"❌ {commodity.upper()} model training failed")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Commodity Price Monitor v1.0**")
st.sidebar.markdown("Real-time monitoring & AI predictions")

# Display last update time
if auto_refresh:
    st.sidebar.markdown(f"*Last updated: {datetime.now().strftime('%H:%M:%S')}*")