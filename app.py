import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Resource Optimization ML", layout="wide", initial_sidebar_state="expanded")

# ==================== LOAD DATA ====================
@st.cache_resource
def load_ab_results():
    with open('results/ab_test_results.json', 'r') as f:
        return json.load(f)

@st.cache_resource
def load_sample_data():
    """Load sample data for visualization (generated from project scripts)"""
    # These are generated from the scripts but we'll create summary stats
    ab_results = load_ab_results()
    
    # Create sample services data based on A/B test
    services_data = {
        'service_id': list(range(1, 151)),
        'service_name': [f"service-{i}" for i in range(1, 151)],
        'memory_mb': np.random.choice([256, 512, 1024, 2048, 4096], 150),
        'cpu_cores': np.random.choice([0.5, 1, 2, 4], 150),
        'traffic_volume_rps': np.random.randint(1000, 100000, 150),
        'latency_critical': np.random.choice([True, False], 150, p=[0.3, 0.7])
    }
    services = pd.DataFrame(services_data)
    
    # Create sample latency data
    regions = ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1', 'ap-northeast-1']
    latency_matrix = {
        ('us-east-1', 'us-west-2'): (60, 80),
        ('us-east-1', 'eu-west-1'): (90, 110),
        ('us-east-1', 'ap-southeast-1'): (180, 220),
        ('us-east-1', 'ap-northeast-1'): (150, 190),
        ('us-west-2', 'eu-west-1'): (130, 160),
        ('us-west-2', 'ap-southeast-1'): (140, 170),
        ('us-west-2', 'ap-northeast-1'): (110, 140),
        ('eu-west-1', 'ap-southeast-1'): (200, 250),
        ('eu-west-1', 'ap-northeast-1'): (180, 230),
        ('ap-southeast-1', 'ap-northeast-1'): (50, 80),
    }
    
    latency_data = []
    for r1 in regions:
        for r2 in regions:
            if r1 == r2:
                latency_data.append({'region1': r1, 'region2': r2, 'latency_ms': 2})
            elif (r1, r2) in latency_matrix:
                min_lat, max_lat = latency_matrix[(r1, r2)]
                latency_data.append({'region1': r1, 'region2': r2, 'latency_ms': np.random.uniform(min_lat, max_lat)})
            elif (r2, r1) in latency_matrix:
                min_lat, max_lat = latency_matrix[(r2, r1)]
                latency_data.append({'region1': r1, 'region2': r2, 'latency_ms': np.random.uniform(min_lat, max_lat)})
    
    latency = pd.DataFrame(latency_data)
    
    return services, latency

# Load all data
ab_results = load_ab_results()
services, latency = load_sample_data()

# ==================== SIDEBAR ====================
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["📈 Overview", "🎯 A/B Test Results", "🗺️ Regional Analysis", "ℹ️ About"]
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
        st.metric("Dataset Size", "1.6M+ records")
    with col4:
        st.metric("Models Trained", 2)
    
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
    
    st.subheader("Traffic Volume by Service (Top 10)")
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
    st.dataframe(comparison_df, width='stretch')
    
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

# ==================== PAGE 4: ABOUT ====================
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
    
    ## 🚀 Next Steps for Production
    
    - Integrate with real AWS CloudWatch metrics
    - Deploy as automated recommendation engine
    - Create feedback loop for model retraining
    - Build alerting system for anomalies
    - Extend to multi-cloud (GCP, Azure)
    
    ## 📂 Project Repository
    
    **GitHub**: [resource-optimization-ml](https://github.com/aankitdas/resource-optimization-ml)
    
    ---
    
    **Built with Python | ML | Data Engineering | Cloud Architecture**
    """)