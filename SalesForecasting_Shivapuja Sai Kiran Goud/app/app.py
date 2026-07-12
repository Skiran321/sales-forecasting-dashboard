"""
Sales Forecasting & Demand Intelligence Dashboard
Internship Project - Week 3 & 4
Shivapuja Sai Kiran Goud

4-page Streamlit app:
  1. Sales Overview
  2. Forecast Explorer
  3. Anomaly Report
  4. Product Demand Segments

Run with: streamlit run app.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")



@st.cache_data
def load_data():
    df = pd.read_csv('train.csv', encoding='latin1')
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y')
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Quarter'] = df['Order Date'].dt.quarter
    return df


def season_num(m):
    if m in [12, 1, 2]:
        return 0
    elif m in [3, 4, 5]:
        return 1
    elif m in [6, 7, 8]:
        return 2
    else:
        return 3


FEATURE_COLS = ['lag1', 'lag2', 'lag3', 'rolling_mean3', 'Month', 'Quarter', 'Season']


def build_monthly(sub_df, freq_min_date, freq_max_date):
    m = sub_df.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum()
    m = m.reindex(pd.date_range(freq_min_date, freq_max_date, freq='ME')).fillna(0)
    return m


def make_features(monthly_series):
    d = pd.DataFrame({'Sales': monthly_series})
    d['lag1'] = d['Sales'].shift(1)
    d['lag2'] = d['Sales'].shift(2)
    d['lag3'] = d['Sales'].shift(3)
    d['rolling_mean3'] = d['Sales'].shift(1).rolling(3).mean()
    d['Month'] = d.index.month
    d['Quarter'] = d.index.quarter
    d['Season'] = d['Month'].apply(season_num)
    return d.dropna()


@st.cache_data
def forecast_with_xgboost(monthly_series, horizon):
    """Train on all but last 3 months, evaluate on them, then forecast `horizon`
    months into the future using the same recursive lag approach as the notebook."""
    d_feat = make_features(monthly_series)
    train = d_feat.iloc[:-3]
    test = d_feat.iloc[-3:]

    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=1)
    model.fit(train[FEATURE_COLS], train['Sales'])

  
    test_pred = model.predict(test[FEATURE_COLS])
    mae = mean_absolute_error(test['Sales'], test_pred)
    rmse = np.sqrt(mean_squared_error(test['Sales'], test_pred))

    
    full_feat = make_features(monthly_series)
    full_model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=1)
    full_model.fit(full_feat[FEATURE_COLS], full_feat['Sales'])

    history = monthly_series.tolist()
    last_date = monthly_series.index[-1]
    preds = []
    for step in range(horizon):
        next_date = last_date + pd.offsets.MonthEnd(step + 1)
        row = pd.DataFrame([[history[-1], history[-2], history[-3], np.mean(history[-3:]),
                              next_date.month, next_date.quarter, season_num(next_date.month)]],
                            columns=FEATURE_COLS)
        p = full_model.predict(row)[0]
        preds.append(p)
        history.append(p)

    idx = pd.date_range(last_date + pd.offsets.MonthEnd(1), periods=horizon, freq='ME')
    return pd.Series(preds, index=idx), mae, rmse


@st.cache_data
def run_anomaly_detection(_df):
    weekly = _df.groupby(pd.Grouper(key='Order Date', freq='W'))['Sales'].sum()
    weekly = weekly.asfreq('W').fillna(0).to_frame('Sales')

    iso = IsolationForest(contamination=0.06, random_state=42)
    weekly['iso_anomaly'] = iso.fit_predict(weekly[['Sales']])

    weekly['rolling_mean'] = weekly['Sales'].rolling(8, center=True, min_periods=1).mean()
    weekly['rolling_std'] = weekly['Sales'].rolling(8, center=True, min_periods=1).std()
    weekly['zscore'] = (weekly['Sales'] - weekly['rolling_mean']) / weekly['rolling_std']
    weekly['z_anomaly'] = weekly['zscore'].abs() > 2

    return weekly


@st.cache_data
def run_segmentation(_df):
    rows = []
    for sc in _df['Sub-Category'].unique():
        sub = _df[_df['Sub-Category'] == sc]
        total_sales = sub['Sales'].sum()
        avg_order_value = sub['Sales'].mean()

        sc_monthly = build_monthly(sub, _df['Order Date'].min(), _df['Order Date'].max())
        volatility = sc_monthly.std()

        yearly = sub.groupby(sub['Order Date'].dt.year)['Sales'].sum().sort_index()
        growth_rate = (yearly.iloc[-1] - yearly.iloc[0]) / yearly.iloc[0] * 100 if yearly.iloc[0] > 0 else 0

        rows.append({'Sub-Category': sc, 'total_sales': total_sales, 'growth_rate': growth_rate,
                     'volatility': volatility, 'avg_order_value': avg_order_value})

    seg_df = pd.DataFrame(rows).set_index('Sub-Category')
    features = ['total_sales', 'growth_rate', 'volatility', 'avg_order_value']
    X_scaled = StandardScaler().fit_transform(seg_df[features].values)

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    seg_df['cluster'] = km.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    seg_df['pca1'], seg_df['pca2'] = coords[:, 0], coords[:, 1]

    cluster_names = {}
    means = seg_df.groupby('cluster')[features].mean()
    for c in means.index:
        row = means.loc[c]
        if row['growth_rate'] > 200:
            cluster_names[c] = "Growing Demand (explosive)"
        elif row['growth_rate'] < 0:
            cluster_names[c] = "Declining Demand"
        elif row['total_sales'] > seg_df['total_sales'].median():
            cluster_names[c] = "High Volume, Growing Demand"
        else:
            cluster_names[c] = "Low Volume, Stable Demand"
    seg_df['cluster_label'] = seg_df['cluster'].map(cluster_names)
    return seg_df


df = load_data()

st.sidebar.title("Sales Forecasting Dashboard")
page = st.sidebar.radio("Go to", ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"])
st.sidebar.markdown("---")
st.sidebar.caption("Superstore Sales dataset | 2015-2018 | XGBoost-based forecasting")


# ---------------------------------------------------------------------------
# PAGE 1 - Sales Overview
# ---------------------------------------------------------------------------
if page == "Sales Overview":
    st.title("Sales Overview Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", f"₹{df['Sales'].sum():,.0f}")
    col2.metric("Total Orders", f"{df['Order ID'].nunique():,}")
    col3.metric("Date Range", f"{df['Order Date'].min().year} - {df['Order Date'].max().year}")

    st.subheader("Total Sales by Year")
    yearly = df.groupby('Year')['Sales'].sum().reset_index()
    fig = px.bar(yearly, x='Year', y='Sales', text_auto='.2s')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Sales Trend")
    monthly = df.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum().reset_index()
    fig2 = px.line(monthly, x='Order Date', y='Sales', markers=True)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Sales by Region and Category")
    col_a, col_b = st.columns(2)
    with col_a:
        region_filter = st.multiselect("Filter by Region", df['Region'].unique(), default=list(df['Region'].unique()))
    with col_b:
        category_filter = st.multiselect("Filter by Category", df['Category'].unique(), default=list(df['Category'].unique()))

    filtered = df[df['Region'].isin(region_filter) & df['Category'].isin(category_filter)]
    grouped = filtered.groupby(['Region', 'Category'])['Sales'].sum().reset_index()
    fig3 = px.bar(grouped, x='Region', y='Sales', color='Category', barmode='group')
    st.plotly_chart(fig3, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 2 - Forecast Explorer
# ---------------------------------------------------------------------------
elif page == "Forecast Explorer":
    st.title("Forecast Explorer")

    dim_type = st.selectbox("Select dimension", ["Category", "Region"])
    if dim_type == "Category":
        dim_value = st.selectbox("Select value", sorted(df['Category'].unique()))
        subset = df[df['Category'] == dim_value]
    else:
        dim_value = st.selectbox("Select value", sorted(df['Region'].unique()))
        subset = df[df['Region'] == dim_value]

    horizon = st.select_slider("Forecast horizon (months ahead)", options=[1, 2, 3], value=3)

    monthly_series = build_monthly(subset, df['Order Date'].min(), df['Order Date'].max())

    with st.spinner("Training model and generating forecast..."):
        preds, mae, rmse = forecast_with_xgboost(monthly_series, horizon)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly_series.index, monthly_series.values, label='Historical Sales')
    ax.plot(preds.index, preds.values, marker='o', color='orange', label='Forecast')
    ax.set_title(f"{dim_value} — {horizon}-Month Forecast (XGBoost)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Forecast values")
    forecast_table = pd.DataFrame({'Month': preds.index.strftime('%b %Y'), 'Forecasted Sales': preds.values.round(0)})
    st.table(forecast_table)

    st.subheader("Model accuracy (backtested on last 3 known months)")
    col1, col2 = st.columns(2)
    col1.metric("MAE", f"₹{mae:,.0f}")
    col2.metric("RMSE", f"₹{rmse:,.0f}")


# ---------------------------------------------------------------------------
# PAGE 3 - Anomaly Report
# ---------------------------------------------------------------------------
elif page == "Anomaly Report":
    st.title("Anomaly Report")
    st.caption("Isolation Forest flags statistically rare weeks; Z-score flags sharp local deviations from the rolling trend.")

    weekly = run_anomaly_detection(df)
    iso_anomalies = weekly[weekly['iso_anomaly'] == -1]
    z_anomalies = weekly[weekly['z_anomaly']]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(weekly.index, weekly['Sales'], label='Weekly Sales', color='steelblue')
    ax.scatter(iso_anomalies.index, iso_anomalies['Sales'], color='red', label='Isolation Forest anomaly', zorder=5, s=60)
    ax.scatter(z_anomalies.index, z_anomalies['Sales'], color='orange', marker='x', label='Z-score anomaly', zorder=5, s=80)
    ax.legend()
    ax.set_title("Weekly Sales with Detected Anomalies")
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Isolation Forest ({len(iso_anomalies)} weeks)")
        st.dataframe(iso_anomalies[['Sales']].reset_index().rename(columns={'Order Date': 'Week'}))
    with col2:
        st.subheader(f"Z-score method ({len(z_anomalies)} weeks)")
        st.dataframe(z_anomalies[['Sales', 'zscore']].reset_index().rename(columns={'Order Date': 'Week'}).round(2))


# ---------------------------------------------------------------------------
# PAGE 4 - Product Demand Segments
# ---------------------------------------------------------------------------
elif page == "Product Demand Segments":
    st.title("Product Demand Segments")

    seg_df = run_segmentation(df)

    fig = px.scatter(seg_df.reset_index(), x='pca1', y='pca2', color='cluster_label',
                      text='Sub-Category', size='total_sales',
                      title="Product Sub-Categories by Demand Cluster")
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sub-categories by cluster")
    display_df = seg_df[['cluster_label', 'total_sales', 'growth_rate', 'volatility', 'avg_order_value']].round(1)
    display_df.columns = ['Cluster', 'Total Sales', 'Growth Rate (%)', 'Volatility', 'Avg Order Value']
    st.dataframe(display_df.sort_values('Cluster'))