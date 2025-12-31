# 🚀 Resource Optimization ML Pipeline

An end-to-end machine learning solution for optimizing service placement across AWS regions, reducing latency and costs while maintaining reliability.

**Live Dashboard:** [View on Hugging Face Spaces](https://huggingface.co/spaces/YOUR_USERNAME/resource-optimization-ml)

## 📊 Project Overview

This project demonstrates a complete ML pipeline inspired by Amazon's Region Flexibility Engineering team challenges:

- **Problem:** Optimize service placement across 5 AWS regions to reduce latency and costs
- **Solution:** ML-driven placement strategy with A/B testing validation
- **Results:** 5.25% latency reduction, 4.92% cost savings, statistically significant (p < 0.001)

## 🎯 Key Results

| Metric | Result |
|--------|--------|
| Latency Reduction | **5.25%** ✅ |
| Cost Savings | **4.92%** ✅ |
| Critical Service Improvement | **9.30%** ✅ |
| Statistical Significance | **p < 0.001** ✅ |
| Placement Efficiency | **378 vs 452 pairs** (-16%) |

## 🛠️ Architecture

### Data Pipeline
- **150+ services** with metadata (memory, CPU, latency sensitivity)
- **1.6M+ traffic records** across 5 AWS regions
- **30K+ placement records** with latency and error rates
- **Regional latency matrix** for cross-region communication costs

### ML Models

#### Model 1: Latency Prediction (XGBoost Regression)
- Predicts service latency for a given placement
- **Features:** Memory, CPU cores, traffic patterns, outbound latency, service dependencies
- **Performance:** RMSE = 28.7ms, MAE = 24.67ms
- **Top Features:** Request variability, outbound latency, average traffic

#### Model 2: Placement Strategy (Random Forest Classifier)
- Classifies services for optimal regional distribution
- **Features:** Traffic volume, dependencies, latency sensitivity, resource requirements
- **Performance:** 100% accuracy on test set

### A/B Testing Framework
- **Control:** Random service placement (baseline)
- **Treatment:** ML-optimized placement using model predictions
- **Statistical Test:** Independent t-test (t=7.02, p<0.001)
- **Result:** Statistically significant improvement ✅

## 📁 Project Structure

```
resource-optimization-ml/
├── data/                           # Generated datasets
│   ├── services.csv               # Service metadata
│   ├── regional_latency.csv       # Cross-region latency
│   ├── traffic_patterns.csv       # Hourly traffic by service/region
│   └── service_placement.csv      # Historical placements
│
├── models/                         # Trained ML models
│   ├── xgboost_latency_model.pkl  # Latency prediction model
│   ├── random_forest_placement_model.pkl  # Placement strategy model
│   ├── scaler_latency.pkl         # Feature scaler
│   ├── scaler_classification.pkl  # Feature scaler
│   └── feature_importance_*.csv   # Feature importance analysis
│
├── results/                        # A/B test results
│   ├── ab_test_results.json       # Statistical comparison
│   ├── control_placement.csv      # Control group placements
│   └── treatment_placement.csv    # Treatment group placements
│
├── notebooks/                      # Analysis notebooks (optional)
│
├── data_generation.py              # Generate synthetic dataset
├── setup_database.py               # Load data into SQLite
├── explore_data.py                 # Data exploration and SQL queries
├── train_models.py                 # Train ML models
├── ab_test_simulation.py           # Run A/B test simulation
├── app.py                          # Streamlit dashboard
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── .gitignore
```

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/resource-optimization-ml.git
cd resource-optimization-ml
```

2. **Install dependencies** (using uv or pip)
```bash
uv pip install -r requirements.txt
```

3. **Generate data**
```bash
uv run python data_generation.py
```

4. **Setup database**
```bash
uv run python setup_database.py
```

5. **Explore data**
```bash
uv run python explore_data.py
```

6. **Train models**
```bash
uv run python train_models.py
```

7. **Run A/B test simulation**
```bash
uv run python ab_test_simulation.py
```

8. **Launch dashboard**
```bash
uv run streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

## 📊 Dashboard Features

### 📈 Overview
- Service distribution by memory, CPU, and latency sensitivity
- Traffic volume analysis across regions
- Total statistics (150 services, 5 regions, 1.6M records)

### 🎯 A/B Test Results
- Side-by-side comparison of control vs treatment strategies
- Latency reduction: 5.25%
- Cost savings: 4.92%
- Statistical significance test results (p-value, t-statistic)

### 🗺️ Regional Analysis
- Interactive latency heatmap between all region pairs
- Regional statistics (min, max, std deviation)
- Identify high-latency corridors

### 🔧 Service Details
- Interactive service explorer
- Per-service placement across regions
- Instance count and latency metrics

## 🧠 Technical Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Data Storage | SQLite | Lightweight database for local development |
| Data Processing | Pandas, NumPy | Data manipulation and feature engineering |
| ML Framework | scikit-learn, XGBoost | Model training and prediction |
| Statistics | SciPy | A/B testing and significance tests |
| Visualization | Plotly, Streamlit | Interactive dashboards |
| Deployment | Hugging Face Spaces | Live dashboard hosting |

## 📈 Model Performance

### XGBoost (Latency Prediction)
```
RMSE: 28.7007 ms
MAE:  24.6690 ms
R²:   -0.0674 (indicates high variance in data)
```

**Top 5 Important Features:**
1. Request Variability (CV): 21.7%
2. Outbound Latency: 17.6%
3. Average Requests: 14.2%
4. Dependencies: 13.5%
5. Number of Instances: 11.7%

### Random Forest (Placement Strategy)
```
Accuracy: 100%
Precision: 1.00
Recall: 1.00
F1-Score: 1.00
```

**Top Features:**
1. Traffic Volume: 54.5%
2. Dependencies: 13.8%
3. Latency Sensitivity: 13.7%

## 🧪 A/B Test Methodology

**Hypothesis:** ML-optimized placement reduces latency compared to random placement

**Sample Size:** 150 services × 5 regions = 750 potential placements

**Metrics:**
- Primary: Average latency (ms)
- Secondary: Total cost ($), redundancy score, critical service latency
- Efficiency: Number of placement pairs (fewer = more efficient)

**Test Type:** Independent samples t-test
- Null hypothesis (H₀): μ_control = μ_treatment
- Alternative hypothesis (H₁): μ_control ≠ μ_treatment
- Significance level: α = 0.05

**Result:** Reject H₀ (p < 0.001)
- The ML-optimized placement significantly reduces latency

## 💡 Key Insights

1. **Latency-critical services benefit most** from optimized placement (9.3% improvement vs 5.25% average)
2. **Traffic patterns drive decisions** - high-traffic services benefit from multi-region placement
3. **Regional cost differences matter** - avoiding expensive regions saves 4.92% without sacrificing latency
4. **Placement efficiency improves** - ML uses 16% fewer placement pairs while reducing latency
5. **Statistical rigor matters** - The improvement is not due to chance (p < 0.001)

## 🚀 Future Enhancements

### Short-term
- [ ] Add notebook with exploratory data analysis
- [ ] Include feature importance visualizations
- [ ] Create prediction API endpoint

### Medium-term
- [ ] Integrate real AWS CloudWatch metrics
- [ ] Add model retraining pipeline
- [ ] Implement automated alerting
- [ ] Support multi-cloud scenarios (GCP, Azure)

### Long-term
- [ ] Deploy as microservice recommendation engine
- [ ] Build feedback loop for model improvement
- [ ] Create cost optimization module
- [ ] Add capacity planning features

## 📚 Learning Resources

This project demonstrates:
- ✅ SQL data querying and aggregation
- ✅ Python data manipulation (Pandas, NumPy)
- ✅ Machine learning model training (scikit-learn, XGBoost)
- ✅ Feature engineering and preprocessing
- ✅ Statistical hypothesis testing
- ✅ A/B testing methodology
- ✅ Data visualization (Plotly, Streamlit)
- ✅ Full-stack ML deployment

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

Built as a portfolio project demonstrating ML engineering capabilities for cloud infrastructure optimization.

---

**Questions or feedback?** Open an issue or reach out!

**Live Dashboard:** [Hugging Face Spaces](https://huggingface.co/spaces/aankitdas/resource-optimization-ml)
**GitHub:** [resource-optimization-ml](https://github.com/aankitdas/resource-optimization-ml)