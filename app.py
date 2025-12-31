import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Resource Optimization ML", layout="wide", initial_sidebar_state="expanded")

# ==================== LOAD DATA ====================
@st.cache_resource
def load_data():
    conn = sqlite3.connect('resource_optimization.db')
    
    services = pd.read_sql_query("SELECT * FROM services", conn)
    latency = pd.read_sql_query("SELECT * FROM regional_latency", conn)
    traffic = pd.read_sql_query("SELECT * FROM traffic_patterns", conn)
    placement = pd.read_sql_query("SELECT * FROM service_placement", conn)
    
    conn.close()
    return services, latency, traffic, placement

@st.cache_resource
def load_ab_results():
    with open('results/ab_test_results.json', 'r') as f:
        return json.load(f)

# Load all data
services, latency, traffic, placement = load_data()
ab_results = load_ab_results()

# ==================== SIDEBAR ====================
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["📈 Overview", "🎯 A/B Test Results", "🗺️ Regional Analysis", "🔧 Service Details", "ℹ️ About"]
)

# ==================== PAGE 1: OVERVIEW ====================
if page == "📈 Overview":
    st.title("🚀 Resource Optimization ML Pipeline")
    
    st.markdown("""
    This project demonstrates an **end-to-end ML solution** for optimizing service placement 
    across AWS regions. The goal: reduce latency and costs while maintaining service reliability.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Services", len(services))
    with col2:
        st.metric("AWS Regions", 5)
    with col3:
        st.metric("Placement Records", len(placement))
    with col4:
        st.metric("Traffic Records", f"{len(traffic)/1_000_000:.1f}M")
    
    st.divider()
    
    # Service Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Services by Memory Requirements")
        memory_dist = services['memory_mb'].value_counts().sort_index()
        fig = px.bar(
            x=memory_dist.index, 
            y=memory_dist.values,
            labels={'x': 'Memory (MB)', 'y': 'Count'},
            color=memory_dist.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("Latency Critical vs Non-Critical")
        critical_dist = services['latency_critical'].value_counts()
        fig = px.pie(
            values=critical_dist.values,
            names=['Non-Critical', 'Latency Critical'],
            color_discrete_sequence=['#636EFA', '#EF553B']
        )
        st.plotly_chart(fig, width='stretch')
    
    st.divider()
    
    st.subheader("Traffic Volume by Service")
    top_services = services.nlargest(10, 'traffic_volume_rps')[['service_name', 'traffic_volume_rps']]
    fig = px.bar(
        top_services,
        x='traffic_volume_rps',
        y='service_name',
        orientation='h',
        labels={'traffic_volume_rps': 'Requests/Second', 'service_name': 'Service'},
        color='traffic_volume_rps',
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig, width='stretch')

# ==================== PAGE 2: A/B TEST RESULTS ====================
elif page == "🎯 A/B Test Results":
    st.title("A/B Test: Random vs ML-Optimized Placement")
    
    st.markdown("""
    Comparing a **random placement strategy** (control) against an **ML-optimized strategy** (treatment).
    """)
    
    control = ab_results['control_metrics']
    treatment = ab_results['treatment_metrics']
    improvements = ab_results['improvements']
    sig = ab_results['statistical_significance']
    
    # Key Metrics Comparison
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Latency Reduction",
            f"{improvements['latency_reduction_pct']:.2f}%",
            delta="Lower is better"
        )
    with col2:
        st.metric(
            "Cost Savings",
            f"{improvements['cost_reduction_pct']:.2f}%",
            delta="Lower is better"
        )
    with col3:
        st.metric(
            "Critical Service Latency",
            f"{improvements['critical_latency_reduction_pct']:.2f}%",
            delta="Lower is better"
        )
    with col4:
        is_sig = "✅ YES" if sig['is_significant'] else "❌ NO"
        st.metric(
            "Statistically Significant?",
            is_sig,
            delta=f"p-value: {sig['p_value']:.6f}"
        )
    
    st.divider()
    
    # Detailed Comparison Table
    st.subheader("Detailed Metrics Comparison")
    comparison_data = {
        'Metric': [
            'Average Latency (ms)',
            'Total Cost ($)',
            'Placement Pairs',
            'Redundancy Score',
            'Critical Service Latency (ms)'
        ],
        'Control (Random)': [
            f"{control['avg_latency_ms']:.2f}",
            f"{control['total_cost']:.2f}",
            f"{control['total_placement_pairs']}",
            f"{control['redundancy_score']:.2f}",
            f"{control['critical_services_latency_ms']:.2f}"
        ],
        'Treatment (ML-Optimized)': [
            f"{treatment['avg_latency_ms']:.2f}",
            f"{treatment['total_cost']:.2f}",
            f"{treatment['total_placement_pairs']}",
            f"{treatment['redundancy_score']:.2f}",
            f"{treatment['critical_services_latency_ms']:.2f}"
        ]
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    st.divider()
    
    # Visual Comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Latency Comparison")
        latency_data = {
            'Strategy': ['Control\n(Random)', 'Treatment\n(ML-Optimized)'],
            'Average Latency (ms)': [control['avg_latency_ms'], treatment['avg_latency_ms']]
        }
        fig = px.bar(
            latency_data,
            x='Strategy',
            y='Average Latency (ms)',
            color_discrete_sequence=['#EF553B', '#00CC96'],
            text='Average Latency (ms)'
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("Cost Comparison")
        cost_data = {
            'Strategy': ['Control\n(Random)', 'Treatment\n(ML-Optimized)'],
            'Total Cost ($)': [control['total_cost'], treatment['total_cost']]
        }
        fig = px.bar(
            cost_data,
            x='Strategy',
            y='Total Cost ($)',
            color_discrete_sequence=['#EF553B', '#00CC96'],
            text='Total Cost ($)'
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, width='stretch')
    
    st.divider()
    
    # Statistical Details
    st.subheader("📊 Statistical Significance Test")
    st.write(f"""
    - **Test Type**: Independent t-test
    - **t-statistic**: {sig['t_statistic']:.4f}
    - **p-value**: {sig['p_value']:.10f}
    - **Result**: {'✅ **STATISTICALLY SIGNIFICANT**' if sig['is_significant'] else '❌ Not significant'} (α = 0.05)
    
    *The improvement in latency is statistically significant, meaning it's unlikely to be due to random chance.*
    """)

# ==================== PAGE 3: REGIONAL ANALYSIS ====================
elif page == "🗺️ Regional Analysis":
    st.title("Regional Latency Analysis")
    
    # Convert timestamp
    latency['timestamp'] = pd.to_datetime(latency['timestamp'])
    
    # Latency heatmap
    st.subheader("Average Cross-Region Latency (ms)")
    
    latency_pivot = latency.pivot_table(
        values='latency_ms',
        index='region1',
        columns='region2',
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=latency_pivot.values,
        x=latency_pivot.columns,
        y=latency_pivot.index,
        colorscale='RdYlGn_r',
        text=np.round(latency_pivot.values, 1),
        texttemplate='%{text} ms',
        textfont={"size": 10}
    ))
    fig.update_layout(title="Latency Heatmap", xaxis_title="To Region", yaxis_title="From Region")
    st.plotly_chart(fig, width='stretch')
    
    st.divider()
    
    # Regional statistics
    st.subheader("Regional Statistics")
    
    latency_stats = latency.groupby('region1').agg({
        'latency_ms': ['mean', 'min', 'max', 'std']
    }).round(2)
    latency_stats.columns = ['Avg Latency (ms)', 'Min (ms)', 'Max (ms)', 'Std Dev (ms)']
    
    st.dataframe(latency_stats, width='stretch')

# ==================== PAGE 4: SERVICE DETAILS ====================
elif page == "🔧 Service Details":
    st.title("Service Details Explorer")
    
    # Service selector
    selected_service_name = st.selectbox(
        "Select a service:",
        services['service_name'].sort_values(),
        key='service_selector'
    )
    
    selected_service = services[services['service_name'] == selected_service_name].iloc[0]
    
    st.subheader(f"Service: {selected_service['service_name']}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Memory", f"{selected_service['memory_mb']} MB")
    with col2:
        st.metric("CPU Cores", selected_service['cpu_cores'])
    with col3:
        st.metric("Traffic (RPS)", f"{selected_service['traffic_volume_rps']:,}")
    with col4:
        st.metric("Dependencies", int(selected_service['dependencies']))
    with col5:
        critical_status = "🔴 Critical" if selected_service['latency_critical'] else "🟢 Normal"
        st.metric("Latency Sensitivity", critical_status)
    
    st.divider()
    
    # Service placement across regions
    service_placement = placement[placement['service_id'] == selected_service['service_id']]
    
    if len(service_placement) > 0:
        st.subheader("Placement Across Regions")
        
        placement_summary = service_placement.groupby('region').agg({
            'instances': 'mean',
            'avg_latency_ms': 'mean',
            'error_rate': 'mean'
        }).round(2)
        
        st.dataframe(placement_summary, width='stretch')
        
        # Latency by region
        fig = px.bar(
            placement_summary,
            y='avg_latency_ms',
            labels={'avg_latency_ms': 'Average Latency (ms)', 'region': 'Region'},
            color='avg_latency_ms',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, width='stretch')

# ==================== PAGE 5: ABOUT ====================
elif page == "ℹ️ About":
    st.title("About This Project")
    
    st.markdown("""
    ## 🎯 Problem Statement
    
    Amazon's Region Flexibility Engineering team needs to optimize service placement across 
    AWS regions to:
    - **Reduce latency** for end users
    - **Lower costs** by avoiding expensive regions
    - **Maintain reliability** with appropriate redundancy
    - **Support rapid global expansion**
    
    ## 🛠️ Solution Architecture
    
    ### 1. Data Pipeline
    - **Sources**: Service metadata, traffic patterns, regional latency, placement history
    - **Processing**: SQL queries + Pandas for feature engineering
    - **Scale**: 150+ services, 5 regions, 1.6M+ traffic records
    
    ### 2. ML Models
    
    **Model 1: Latency Prediction (XGBoost)**
    - Predicts service latency for a given placement
    - Features: Memory, CPU, traffic patterns, outbound latency
    - Performance: RMSE = 28.7ms
    
    **Model 2: Placement Strategy (Random Forest)**
    - Classifies services as high/low traffic
    - Determines optimal number of regions per service
    - Accuracy: 100% on test set
    
    ### 3. A/B Testing Framework
    - **Control**: Random service placement (baseline)
    - **Treatment**: ML-optimized placement
    - **Results**: 5.25% latency reduction, 4.92% cost savings, statistically significant (p < 0.001)
    
    ## 📊 Key Metrics
    
    | Metric | Result |
    |--------|--------|
    | Latency Reduction | 5.25% |
    | Cost Savings | 4.92% |
    | Critical Service Improvement | 9.30% |
    | Statistical Significance | p < 0.001 ✅ |
    | Placement Efficiency | 378 vs 452 pairs (-16%) |
    
    ## 💻 Tech Stack
    
    - **Data**: SQLite, Pandas, NumPy
    - **ML**: scikit-learn, XGBoost
    - **Statistics**: SciPy (t-tests, significance)
    - **Visualization**: Plotly, Streamlit
    - **Deployment**: Hugging Face Spaces
    
    ## 📚 How to Use
    
    1. **Overview**: See project summary and data distribution
    2. **A/B Results**: Detailed comparison of strategies with statistical validation
    3. **Regional Analysis**: Explore latency patterns across AWS regions
    4. **Service Details**: Interactive explorer for individual services
    
    ## 🚀 Next Steps for Production
    
    - Integrate with real AWS CloudWatch metrics
    - Deploy as automated recommendation engine
    - Create feedback loop for model retraining
    - Build alerting system for anomalies
    - Extend to multi-cloud (GCP, Azure)
    
    ---
    
    **Built with Python | ML | Data Engineering | Cloud Architecture**
    """)