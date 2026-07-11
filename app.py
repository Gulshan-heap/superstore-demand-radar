# ==============================================================================
# TASK 7: STREAMLIT ENTERPRISE UNIVERSAL-THEME MULTIPAGE DASHBOARD (app.py)
# ==============================================================================
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Page Configuration Setup (Fluid Responsive Layout)
st.set_page_config(page_title="Superstore Demand Radar", page_icon="📊", layout="wide")

# Custom Styling to support high visibility in both Dark and Light System Themes
st.markdown("""
    <style>
    .main-title { font-size: 32px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .sub-title { font-size: 16px; color: #555555; margin-bottom: 25px; }
    .section-header { font-size: 20px; font-weight: bold; color: #1E3A8A; margin-top: 20px; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Superstore Sales & Demand Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Forecasting, anomaly detection, and product demand segmentation for supply chain planning</div>', unsafe_allow_html=True)

# 2. Optimized & Cached Data Loading Engine (With Strict Date Controls)
@st.cache_data
def load_and_clean_data():
    file_path = 'train.csv'
    if not os.path.exists(file_path):
        st.error(f"❌ '{file_path}' not found! Place train.csv in the same folder as app.py.")
        return pd.DataFrame()

    raw_df = pd.read_csv(file_path)
    
    # DATE PARSING FIX: Using the verified strict format matching our notebook discovery
    raw_df['Order Date'] = pd.to_datetime(raw_df['Order Date'], format='%d/%m/%Y', errors='coerce')
    raw_df['Ship Date'] = pd.to_datetime(raw_df['Ship Date'], format='%d/%m/%Y', errors='coerce')
    
    # Drop records containing unparseable dates to protect downstream pipelines
    raw_df = raw_df.dropna(subset=['Order Date'])

    raw_df['Year'] = raw_df['Order Date'].dt.year
    raw_df['Month'] = raw_df['Order Date'].dt.month
    raw_df['Order_Date_Month'] = raw_df['Order Date'].dt.to_period('M').dt.to_timestamp()
    
    # WEEKLY BUCKETING ALIGNMENT FIX: Maps exactly to the notebook's 'W-MON' resampling boundaries
    raw_df['Order_Date_Week'] = raw_df['Order Date'].dt.to_period('W-MON').dt.start_time
    
    raw_df['Days_to_Ship'] = (raw_df['Ship Date'] - raw_df['Order Date']).dt.days
    return raw_df

df = load_and_clean_data()

if df.empty:
    st.stop()

# ------------------------------------------------------------------------------
# MULTIPAGE SIDEBAR NAVIGATION BACKBONE
# ------------------------------------------------------------------------------
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Sales Overview", 
     "Forecast Explorer", 
     "Anomaly Report", 
     "Product Demand Segments"]
)

UNIVERSAL_TEMPLATE = "plotly"

# ==============================================================================
# PAGE 1: SALES OVERVIEW DASHBOARD MODULE
# ==============================================================================
if page == "Sales Overview":
    st.title("Sales Overview")
    st.caption("Annual and monthly sales trends, filterable by category and region.")

    st.markdown("### Filters")
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        sel_category = st.selectbox("Category", ['All Categories'] + list(df['Category'].dropna().unique()))
    with f_col2:
        sel_region = st.selectbox("Region", ['All Regions'] + list(df['Region'].dropna().unique()))

    f_df = df.copy()
    if sel_category != 'All Categories':
        f_df = f_df[f_df['Category'] == sel_category]
    if sel_region != 'All Regions':
        f_df = f_df[f_df['Region'] == sel_region]

    g_col1, g_col2 = st.columns(2)

    with g_col1:
        annual_sales = f_df.groupby('Year')['Sales'].sum().reset_index()
        fig_bar = px.bar(annual_sales, x='Year', y='Sales',
                         title=f"Total Sales by Year ({sel_category} / {sel_region})",
                         labels={'Sales': 'Total Sales ($)', 'Year': 'Year'},
                         template=UNIVERSAL_TEMPLATE)
        fig_bar.update_traces(marker=dict(color='#00E5FF', line=dict(color='#FFFFFF', width=1)))
        fig_bar.update_layout(xaxis=dict(type='category'))
        st.plotly_chart(fig_bar, width="stretch")

    with g_col2:
        monthly_sales = f_df.groupby('Order_Date_Month')['Sales'].sum().reset_index()
        fig_line = px.line(monthly_sales, x='Order_Date_Month', y='Sales',
                          title="Monthly Sales Trend",
                          labels={'Sales': 'Sales ($)', 'Order_Date_Month': 'Month'},
                          template=UNIVERSAL_TEMPLATE)
        fig_line.update_traces(line=dict(color='#FFC107', width=3), mode='lines+markers')
        st.plotly_chart(fig_line, width="stretch")

# ==============================================================================
# PAGE 2: FORECAST EXPLORER MODULE (DARK-THEME FIX ACTIVE)
# ==============================================================================
elif page == "Forecast Explorer":
    st.title("Forecast Explorer")
    st.caption("Select a category or region to view its sales forecast and model accuracy.")

    st.markdown("### Filters")
    in_col1, in_col2 = st.columns(2)
    with in_col1:
        segment_type = st.selectbox("Forecast by:", ["Category", "Region"])
    with in_col2:
        available_options = list(df[segment_type].dropna().unique())
        selected_segment_val = st.selectbox(f"Select {segment_type}:", available_options)

    forecast_months_horizon = st.slider("Forecast horizon (months ahead):", min_value=1, max_value=3, value=3)

    @st.cache_data
    def run_segment_prophet_pipeline(seg_type, val, horizon):
        seg_df = df[df[seg_type] == val]
        ts_base = seg_df.groupby('Order_Date_Month')['Sales'].sum().sort_index().reset_index()
        p_input = ts_base.rename(columns={'Order_Date_Month': 'ds', 'Sales': 'y'})
        
        if len(p_input) < 12:
            return None, None, None, None, None, p_input

        # Split data into train/test splits for Segment-Specific Backtesting
        validation_months = 6
        train_slice = p_input.iloc[:-validation_months].copy()
        test_slice = p_input.iloc[-validation_months:].copy()

        m_val = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m_val.fit(train_slice)
        val_forecast = m_val.predict(test_slice[['ds']])
        
        seg_mae = mean_absolute_error(test_slice['y'], val_forecast['yhat'])
        seg_rmse = np.sqrt(mean_squared_error(test_slice['y'], val_forecast['yhat']))

        # Fit production model on Full historical timeline data
        m_prod = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m_prod.fit(p_input)
        future_frame = m_prod.make_future_dataframe(periods=horizon, freq='MS')
        prod_forecast = m_prod.predict(future_frame)

        return m_prod, prod_forecast, seg_mae, seg_rmse, validation_months, p_input

    with st.spinner("Fitting forecast model..."):
        m_prod, prod_forecast, live_mae, live_rmse, val_len, p_input = run_segment_prophet_pipeline(segment_type, selected_segment_val, forecast_months_horizon)

    if m_prod is not None:
        hist_len = len(p_input)
        hist_part = prod_forecast.iloc[:hist_len]
        fc_plot_part = prod_forecast.iloc[hist_len-1:]

        fig_fc = go.Figure()
        
        # 🎯 FIX: Changed color from 'black' to high-visibility '#29B6F6' (Electric Sky Blue)
        fig_fc.add_trace(go.Scatter(x=p_input['ds'], y=p_input['y'], name='Historical Sales', line=dict(color='#29B6F6', width=2)))
        
        # Future forecast vector track
        fig_fc.add_trace(go.Scatter(x=fc_plot_part['ds'], y=fc_plot_part['yhat'], name='Forecast', line=dict(color='#00E676', width=3, dash='dash')))
        
        # Uncertainty bounds shading envelope
        fig_fc.add_trace(go.Scatter(
            x=list(fc_plot_part['ds']) + list(fc_plot_part['ds'])[::-1],
            y=list(fc_plot_part['yhat_upper']) + list(fc_plot_part['yhat_lower'])[::-1],
            fill='toself', fillcolor='rgba(0, 230, 118, 0.12)', line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", showlegend=True, name="95% Confidence Interval"
        ))

        fig_fc.update_layout(title=f"Next {forecast_months_horizon} Month(s) Forecast — {selected_segment_val}",
                            template=UNIVERSAL_TEMPLATE, xaxis_title="Timeline", yaxis_title="Sales ($)")
        st.plotly_chart(fig_fc, width="stretch")

        st.markdown("---")
        st.markdown(f"#### Model Accuracy — {selected_segment_val}")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(label="MAE", value=f"${live_mae:,.2f}")
        with m_col2:
            st.metric(label="RMSE", value=f"${live_rmse:,.2f}")
            
        future_projections = fc_plot_part.tail(forecast_months_horizon)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        future_projections['ds'] = future_projections['ds'].dt.strftime('%b %Y')
        future_projections.columns = ['Month', 'Forecast ($)', 'Lower Bound ($)', 'Upper Bound ($)']
        st.markdown("**Forecast detail:**")
        st.dataframe(future_projections.set_index('Month').style.format("{:,.2f}"), width="stretch")
    else:
        st.warning("Not enough historical data in this segment to fit a reliable forecast.")

# ==============================================================================
# PAGE 3: ANOMALY REPORT MODULE
# ==============================================================================
elif page == "Anomaly Report":
    st.title("Anomaly Report")
    st.caption("Weekly sales anomalies flagged by Isolation Forest and rolling Z-Score.")

    ts_weekly = df.groupby('Order_Date_Week')['Sales'].sum().reset_index()
    ts_weekly.columns = ['Order_Date_Week', 'Total_Sales']

    st.markdown("### Detection method")
    algo_choice = st.radio("Choose a method:", 
                           ("Both methods (overlap)", "Isolation Forest", "Rolling Z-Score"), 
                           horizontal=True)

    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    ts_weekly['IF_Score'] = iso_forest.fit_predict(ts_weekly[['Total_Sales']])
    ts_weekly['Is_Anomaly_IF'] = ts_weekly['IF_Score'] == -1

    ts_weekly['Rolling_Mean'] = ts_weekly['Total_Sales'].rolling(window=8, min_periods=1).mean()
    ts_weekly['Rolling_Std'] = ts_weekly['Total_Sales'].rolling(window=8, min_periods=1).std().fillna(0)
    ts_weekly['Z_Score'] = (ts_weekly['Total_Sales'] - ts_weekly['Rolling_Mean']) / (ts_weekly['Rolling_Std'] + 1e-5)
    ts_weekly['Is_Anomaly_Z'] = np.abs(ts_weekly['Z_Score']) > 2.0

    if "Isolation Forest" in algo_choice:
        ts_weekly['Active_Anomaly'] = ts_weekly['Is_Anomaly_IF']
        plot_title = "Anomalies — Isolation Forest"
    elif "Rolling Z-Score" in algo_choice:
        ts_weekly['Active_Anomaly'] = ts_weekly['Is_Anomaly_Z']
        plot_title = "Anomalies — Rolling Z-Score"
    else:
        ts_weekly['Active_Anomaly'] = ts_weekly['Is_Anomaly_IF'] & ts_weekly['Is_Anomaly_Z']
        plot_title = "High-Confidence Anomalies (Flagged by Both Methods)"

    fig_anom = go.Figure()
    fig_anom.add_trace(go.Scatter(x=ts_weekly['Order_Date_Week'], y=ts_weekly['Total_Sales'], name='Weekly Sales', line=dict(color='#A0AEC0', width=1.5)))
    
    anomalies_only = ts_weekly[ts_weekly['Active_Anomaly']]
    fig_anom.add_trace(go.Scatter(x=anomalies_only['Order_Date_Week'], y=anomalies_only['Total_Sales'], 
                                  mode='markers', name='Anomaly',
                                  marker=dict(color='#FF1744', size=11, symbol='x', line=dict(width=1.5))))
    
    fig_anom.update_layout(title=plot_title, template=UNIVERSAL_TEMPLATE, xaxis_title="Week", yaxis_title="Sales ($)")
    st.plotly_chart(fig_anom, width="stretch")

    st.markdown("---")
    st.markdown("#### Flagged anomaly weeks")
    
    if not anomalies_only.empty:
        table_out = anomalies_only[['Order_Date_Week', 'Total_Sales']].copy()
        table_out['Order_Date_Week'] = table_out['Order_Date_Week'].dt.strftime('%d %b %Y')
        table_out.columns = ['Week', 'Sales ($)']
        st.dataframe(table_out.set_index('Week').style.format({'Sales ($)': '{:,.2f}'}), width="stretch")
    else:
        st.info("No anomalies found under the selected method.")

# ==============================================================================
# PAGE 4: PRODUCT DEMAND SEGMENTS MODULE
# ==============================================================================
elif page == "Product Demand Segments":
    st.title("Product Demand Segments")
    st.caption("K-Means clusters of product sub-categories with recommended stocking strategy per cluster.")

    @st.cache_data
    def compute_live_kmeans_segmentation(_input_df):
        subcat_base = _input_df.groupby('Sub-Category').agg(
            Total_Volume=('Quantity' if 'Quantity' in _input_df.columns else 'Sales', 'sum'),
            Avg_Order_Value=('Sales', 'mean')
        ).reset_index()

        monthly_subcat_sales = _input_df.groupby(['Sub-Category', 'Year', 'Month'])['Sales'].sum().reset_index()
        subcat_volatility = monthly_subcat_sales.groupby('Sub-Category')['Sales'].std().fillna(0).reset_index()
        subcat_volatility.columns = ['Sub-Category', 'Sales_Volatility']

        yoy_sales = _input_df.groupby(['Sub-Category', 'Year'])['Sales'].sum().unstack().fillna(0)
        
        years_sorted = sorted(yoy_sales.columns)
        latest_yr, prev_yr = years_sorted[-1], years_sorted[-2]
        
        yoy_sales['Growth_Rate'] = (yoy_sales[latest_yr] - yoy_sales[prev_yr]) / (yoy_sales[prev_yr] + 1e-5)
        subcat_growth = yoy_sales['Growth_Rate'].reset_index()

        cluster_base_frame = subcat_base.merge(subcat_volatility, on='Sub-Category').merge(subcat_growth, on='Sub-Category')
        
        features_to_scale = ['Total_Volume', 'Avg_Order_Value', 'Sales_Volatility', 'Growth_Rate']
        scaler = StandardScaler()
        scaled_matrix = scaler.fit_transform(cluster_base_frame[features_to_scale])
        
        kmeans_engine = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
        cluster_base_frame['Cluster_Label'] = kmeans_engine.fit_predict(scaled_matrix).astype(str)
        
        return cluster_base_frame

    with st.spinner("Running K-Means clustering..."):
        cluster_df = compute_live_kmeans_segmentation(df)

    fig_cluster = px.scatter(cluster_df, x='Total_Volume', y='Sales_Volatility', color='Cluster_Label',
                             text='Sub-Category', title='Sub-Category Clusters',
                             labels={'Total_Volume': 'Total Sales Volume', 'Sales_Volatility': 'Sales Volatility ($)', 'Cluster_Label': 'Cluster'},
                             color_discrete_sequence=['#FF1744', '#00E676', '#00E5FF', '#FFC107'],
                             template=UNIVERSAL_TEMPLATE)
    
    fig_cluster.update_traces(marker=dict(size=14, line=dict(color='#FFFFFF', width=1)), textposition='top center')
    st.plotly_chart(fig_cluster, width="stretch")

    st.markdown("---")
    st.markdown("#### Stocking strategy by cluster")
    
    strat_col1, strat_col2 = st.columns([1, 2])
    
    with strat_col1:
        st.markdown("**Sub-categories by cluster**")
        table_cluster = cluster_df[['Sub-Category', 'Cluster_Label']].sort_values(by='Cluster_Label').copy()
        table_cluster.columns = ['Sub-Category', 'Cluster']
        st.dataframe(table_cluster.set_index('Sub-Category'), width="stretch")
        
    with strat_col2:
        st.markdown("**Recommended stocking approach**")
        st.markdown("""
        * **Cluster 0 — Low Volume, High Volatility:** Just-in-time procurement. Keep on-site stock minimal; fulfill via fast-turnaround supplier orders.
        * **Cluster 1 — Growing Demand, High Value:** Increase safety stock 15–20% ahead of peak quarters to keep pace with demand.
        * **Cluster 2 — High Volume, Stable Demand:** Automated fixed reorder points. Buy in bulk for better freight rates.
        * **Cluster 3 — Low Volume, Declining Demand:** Reduce inventory commitment; consider third-party drop-shipping.
        """)
