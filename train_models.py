import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Training ML Models\n")

# ==================== LOAD DATA ====================
print("="*70)
print("Loading Data from Database")
print("="*70)

conn = sqlite3.connect('resource_optimization.db')

# Load all tables
services = pd.read_sql_query("SELECT * FROM services", conn)
latency = pd.read_sql_query("SELECT * FROM regional_latency", conn)
traffic = pd.read_sql_query("SELECT * FROM traffic_patterns", conn)
placement = pd.read_sql_query("SELECT * FROM service_placement", conn)

print(f"Loaded {len(services)} services")
print(f"Loaded {len(latency)} latency records")
print(f"Loaded {len(traffic)} traffic records")
print(f"Loaded {len(placement)} placement records\n")

# ==================== FEATURE ENGINEERING ====================
print("="*70)
print("Feature Engineering")
print("="*70)

# Create a feature matrix from placement data
placement['timestamp'] = pd.to_datetime(placement['timestamp'])
traffic['timestamp'] = pd.to_datetime(traffic['timestamp'])

# Aggregate traffic by service and region
traffic_agg = traffic.groupby(['service_id', 'region']).agg({
    'requests': ['mean', 'std', 'max'],
    'hour': 'count'  # number of hours in dataset
}).reset_index()

traffic_agg.columns = ['service_id', 'region', 'avg_requests', 'std_requests', 'max_requests', 'num_hours']
traffic_agg['cv_requests'] = traffic_agg['std_requests'] / (traffic_agg['avg_requests'] + 1)  # coefficient of variation

# Aggregate latency by region pair
latency_agg = latency.groupby(['region1', 'region2']).agg({
    'latency_ms': ['mean', 'std']
}).reset_index()
latency_agg.columns = ['region1', 'region2', 'avg_latency', 'std_latency']

# Create training dataset for MODEL 1 (Latency Prediction)
print("\nBuilding training dataset for latency prediction...")

# Merge placement with service info and traffic
training_data = placement.merge(services[['service_id', 'memory_mb', 'cpu_cores', 'latency_critical', 'dependencies']], 
                                 on='service_id', how='left')
training_data = training_data.merge(traffic_agg, 
                                     left_on=['service_id', 'region'], 
                                     right_on=['service_id', 'region'], 
                                     how='left')

# Merge with latency info (use region to all other regions as features)
# For simplicity, we'll add the average latency from this region to all others
region_latency_avg = latency.groupby('region1')['latency_ms'].mean().reset_index()
region_latency_avg.columns = ['region', 'avg_outbound_latency']
training_data = training_data.merge(region_latency_avg, on='region', how='left')

# Fill missing values
training_data = training_data.fillna(0)

print(f"Created training dataset with {len(training_data)} rows and {training_data.shape[1]} columns")

# ==================== MODEL 1: LATENCY PREDICTION (XGBoost Regression) ====================
print("\n" + "="*70)
print("MODEL 1: LATENCY PREDICTION (XGBoost Regression)")
print("="*70)

# Features for latency prediction
feature_cols_latency = ['memory_mb', 'cpu_cores', 'dependencies', 'avg_requests', 
                        'std_requests', 'max_requests', 'cv_requests', 'avg_outbound_latency', 'instances']

X_latency = training_data[feature_cols_latency].fillna(0)
y_latency = training_data['avg_latency_ms']

# Remove any rows with NaN or infinite values
mask = ~(X_latency.isna().any(axis=1) | np.isinf(X_latency.values).any(axis=1) | y_latency.isna())
X_latency = X_latency[mask]
y_latency = y_latency[mask]

X_train_lat, X_test_lat, y_train_lat, y_test_lat = train_test_split(
    X_latency, y_latency, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train_lat)}, Test set: {len(X_test_lat)}")

# Scale features
scaler_latency = StandardScaler()
X_train_lat_scaled = scaler_latency.fit_transform(X_train_lat)
X_test_lat_scaled = scaler_latency.transform(X_test_lat)

# Train XGBoost
model_xgb = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    verbosity=0
)

model_xgb.fit(X_train_lat_scaled, y_train_lat)

# Evaluate
y_pred_lat = model_xgb.predict(X_test_lat_scaled)
mse = mean_squared_error(y_test_lat, y_pred_lat)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_lat, y_pred_lat)
r2 = r2_score(y_test_lat, y_pred_lat)

print(f"\nModel trained!")
print(f"   RMSE: {rmse:.4f} ms")
print(f"   MAE:  {mae:.4f} ms")
print(f"   R²:   {r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols_latency,
    'importance': model_xgb.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 5 Important Features:")
print(feature_importance.head())

# Save model
joblib.dump(model_xgb, 'models/xgboost_latency_model.pkl')
joblib.dump(scaler_latency, 'models/scaler_latency.pkl')
print(f"Saved to models/xgboost_latency_model.pkl")

# ==================== MODEL 2: PLACEMENT STRATEGY (Classification) ====================
print("\n" + "="*70)
print("MODEL 2: PLACEMENT STRATEGY (Classification)")
print("="*70)

# Create classification target: single-region (0) vs multi-region (1)
placement_counts = placement.groupby('service_id')['region'].nunique().reset_index()
placement_counts.columns = ['service_id', 'num_regions']
placement_counts['strategy'] = (placement_counts['num_regions'] > 1).astype(int)

# Merge with service features
classification_data = services.merge(placement_counts, on='service_id', how='left')

X_class = classification_data[['memory_mb', 'cpu_cores', 'latency_critical', 'traffic_volume_rps', 'dependencies']]
y_class = classification_data['strategy']

print(f"Class distribution: {y_class.value_counts().to_dict()}")

# Check if we have both classes
if len(y_class.unique()) > 1:
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    print(f"Training set: {len(X_train_cls)}, Test set: {len(X_test_cls)}")
    
    # Scale features
    scaler_class = StandardScaler()
    X_train_cls_scaled = scaler_class.fit_transform(X_train_cls)
    X_test_cls_scaled = scaler_class.transform(X_test_cls)
    
    # Train classifier
    model_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        class_weight='balanced'
    )
    
    model_rf.fit(X_train_cls_scaled, y_train_cls)
    
    # Evaluate
    y_pred_cls = model_rf.predict(X_test_cls_scaled)
    accuracy = accuracy_score(y_test_cls, y_pred_cls)
    
    print(f"\nModel trained!")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test_cls, y_pred_cls, labels=[0, 1], target_names=['Single-Region', 'Multi-Region']))
else:
    print(f"\nWARNING: Only one class found in data (all services are multi-region)")
    print(f"   Creating a synthetic binary target for demonstration...")
    
    # Create synthetic target based on threshold of traffic volume
    threshold = X_class['traffic_volume_rps'].median()
    y_class = (X_class['traffic_volume_rps'] > threshold).astype(int)
    
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    print(f"New class distribution (high vs low traffic): {y_class.value_counts().to_dict()}")
    print(f"Training set: {len(X_train_cls)}, Test set: {len(X_test_cls)}")
    
    # Scale features
    scaler_class = StandardScaler()
    X_train_cls_scaled = scaler_class.fit_transform(X_train_cls)
    X_test_cls_scaled = scaler_class.transform(X_test_cls)
    
    # Train classifier
    model_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        class_weight='balanced'
    )
    
    model_rf.fit(X_train_cls_scaled, y_train_cls)
    
    # Evaluate
    y_pred_cls = model_rf.predict(X_test_cls_scaled)
    accuracy = accuracy_score(y_test_cls, y_pred_cls)
    
    print(f"\nModel trained!")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report (High vs Low Traffic Services):")
    print(classification_report(y_test_cls, y_pred_cls, labels=[0, 1], target_names=['Low Traffic', 'High Traffic']))

# Feature importance
feature_importance_cls = pd.DataFrame({
    'feature': X_class.columns,
    'importance': model_rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop Features for Placement Strategy:")
print(feature_importance_cls)

# Save model
joblib.dump(model_rf, 'models/random_forest_placement_model.pkl')
joblib.dump(scaler_class, 'models/scaler_classification.pkl')
print(f"Saved to models/random_forest_placement_model.pkl")

# ==================== SAVE FEATURE IMPORTANCE ====================
print("\n" + "="*70)
print("Saving Feature Importance")
print("="*70)

feature_importance.to_csv('models/feature_importance_latency.csv', index=False)
feature_importance_cls.to_csv('models/feature_importance_placement.csv', index=False)
print("Feature importance saved")

# ==================== SUMMARY ====================
print("\n" + "="*70)
print("MODEL TRAINING COMPLETE!")
print("="*70)
print(f"\nModels saved in 'models/' folder:")
print(f"   • xgboost_latency_model.pkl")
print(f"   • random_forest_placement_model.pkl")
print(f"   • scaler_latency.pkl")
print(f"   • scaler_classification.pkl")
print(f"   • feature_importance_latency.csv")
print(f"   • feature_importance_placement.csv")

print(f"\nModel Performance Summary:")
print(f"   XGBoost (Latency Prediction)")
print(f"      - RMSE: {rmse:.4f} ms")
print(f"      - R²: {r2:.4f}")
print(f"   Random Forest (Placement Strategy)")
print(f"      - Accuracy: {accuracy:.4f}")


conn.close()