---
title: Resource Optimization ML Pipeline
emoji: 🔥
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: latest
app_file: app.py
pinned: false
---

# Resource Optimization ML Pipeline

A data-driven approach to optimizing service placement across cloud regions, reducing latency and infrastructure costs through machine learning.

**Live Dashboard:** https://huggingface.co/spaces/aankitdas/resource-optimization-ml

## Problem

When Amazon scales infrastructure globally across multiple AWS regions, teams face a critical decision: which services should run in which regions? 

The naive approach (random placement) is inefficient:
- Services get placed in expensive regions unnecessarily
- Cross-region communication adds latency
- Over-provisioning of resources to ensure redundancy
- No data-driven strategy for placement decisions

This project tackles that problem: **given service characteristics and regional latency patterns, can we predict optimal placement that reduces latency and costs?**

## Solution

I built an ML-powered recommendation system that:

1. **Analyzes service characteristics** - memory, CPU, traffic volume, latency sensitivity
2. **Models regional latency** - how long it takes to communicate between regions
3. **Predicts placement impact** - what happens to latency if we place a service in region X vs Y
4. **Compares strategies** - random placement vs ML-optimized placement through A/B testing

## Results

The ML-optimized strategy outperforms random placement:

- **5.25% latency reduction** - services respond faster to users
- **4.92% cost savings** - avoided expensive regions where possible
- **9.30% improvement for critical services** - latency-sensitive workloads benefit most
- **Statistical significance** - improvements are not due to chance (p < 0.001)
- **16% fewer placements** - more efficient resource usage

## Technical Approach

### Data Pipeline
- Generated 150 synthetic services with realistic attributes
- Created 1.6M+ traffic records across 5 regions over 90 days
- Modeled cross-region latency patterns based on real AWS geography
- Stored everything in SQLite for easy SQL querying

### Machine Learning

**Model 1: Latency Prediction (XGBoost Regressor)**
- Predicts service latency given placement characteristics
- Input: service memory/CPU, traffic patterns, outbound latency, dependencies
- Output: expected latency in milliseconds
- Performance: RMSE=28.7ms

**Model 2: Placement Strategy (Random Forest Classifier)**
- Determines if a service should be single-region or multi-region
- Input: traffic volume, dependencies, resource requirements
- Output: optimal placement strategy
- Performance: 100% accuracy on test set

### A/B Testing

To validate the ML approach:
- **Control**: randomly place services across 2-4 regions
- **Treatment**: use ML models to recommend optimal placement
- **Test**: independent t-test on latency samples (t=7.02, p<0.001)
- **Conclusion**: ML strategy is statistically significantly better

## How to Use the Dashboard

**Overview** - See service distribution across memory tiers and latency sensitivity. Top services by traffic volume.

**A/B Test Results** - The core finding. Side-by-side comparison of random vs ML-optimized placement with metrics and statistical test results.

**Regional Analysis** - Latency heatmap showing communication costs between regions. Higher latency regions are avoided when possible.

## Project Structure

```
├── data_generation.py         # Generate synthetic services, traffic, latency data
├── setup_database.py          # Load CSVs into SQLite
├── train_models.py            # Train XGBoost and Random Forest models
├── ab_test_simulation.py      # Run A/B test and save results
├── app.py                     # Streamlit dashboard
├── results/
│   └── ab_test_results.json   # A/B test metrics and statistics
└── requirements.txt           # Python dependencies
```

## Technology Stack

- **Data Processing**: Python, Pandas, NumPy, SQLite
- **Machine Learning**: scikit-learn, XGBoost
- **Statistics**: SciPy (hypothesis testing)
- **Visualization**: Plotly, Streamlit
- **Deployment**: Docker, Hugging Face Spaces, GitHub Actions

## Key Insights

1. **Traffic patterns matter most** - Services with high, variable traffic benefit most from multi-region placement

2. **Latency-critical services are placement-sensitive** - A few milliseconds of additional latency can degrade user experience for these workloads

3. **Regional cost differences are significant** - Some regions are 80% more expensive than others. ML avoids them when latency permits

4. **Efficiency and performance can both improve** - ML uses fewer total placements while reducing latency

5. **Statistical rigor matters** - Raw improvements mean nothing without significance testing

## Running Locally

```bash
# Generate data
python data_generation.py

# Setup database
python setup_database.py

# Train models
python train_models.py

# Run A/B test
python ab_test_simulation.py

# Launch dashboard
streamlit run app.py
```

## What This Demonstrates

- SQL data analysis and aggregation
- Python data manipulation and feature engineering
- Machine learning model training and evaluation
- Statistical hypothesis testing and A/B testing methodology
- End-to-end data product development (from data to dashboard)
- Production deployment with Docker and GitHub Actions

## Repository

https://github.com/aankitdas/resource-optimization-ml