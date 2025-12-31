import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

fake = Faker()

print("Starting Data Generation...")

# ==================== PART 1: Generate Services ====================
print("\nGenerating Services Data...")

services_data = []
service_templates = [
    "auth", "cache", "database", "api", "notification",
    "search", "recommendation", "payment", "inventory", "profile",
    "order", "analytics", "logging", "metrics", "config",
    "gateway", "queue", "processor", "manager", "service",
    "worker", "scheduler", "validator", "router", "balancer"
]

# Generate 150 services by combining templates
service_names = []
for i in range(6):
    for template in service_templates:
        service_names.append(f"{template}-service-{i+1}")

for i, name in enumerate(service_names, start=1):
    services_data.append({
        'service_id': i,
        'service_name': name,
        'memory_mb': random.choice([256, 512, 1024, 2048, 4096]),
        'cpu_cores': random.choice([0.5, 1, 2, 4]),
        'latency_critical': random.choice([True, False]),
        'traffic_volume_rps': random.randint(1000, 100000),  # requests per second
        'dependencies': random.randint(0, 5)  # how many other services it depends on
    })

services_df = pd.DataFrame(services_data)
services_df.to_csv('data/services.csv', index=False)
print(f"Generated {len(services_df)} services")
print(services_df.head())

# ==================== PART 2: Generate Regional Latency ====================
print("\nGenerating Regional Latency Data...")

regions = ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1', 'ap-northeast-1']
latency_data = []

# Create latency matrix (some regions are closer than others)
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

# Generate latency measurements over time
start_date = datetime(2024, 1, 1)
for days in range(90):  # 3 months
    timestamp = start_date + timedelta(days=days)
    
    for region1 in regions:
        for region2 in regions:
            if region1 == region2:
                latency_data.append({
                    'region1': region1,
                    'region2': region2,
                    'latency_ms': random.gauss(2, 0.5),  # same region: ~2ms
                    'timestamp': timestamp
                })
            elif (region1, region2) in latency_matrix:
                min_lat, max_lat = latency_matrix[(region1, region2)]
                base_latency = np.random.uniform(min_lat, max_lat)
                # Add some noise
                latency = base_latency + random.gauss(0, 5)
                latency_data.append({
                    'region1': region1,
                    'region2': region2,
                    'latency_ms': max(latency, 1),  # ensure positive
                    'timestamp': timestamp
                })
            elif (region2, region1) in latency_matrix:
                min_lat, max_lat = latency_matrix[(region2, region1)]
                base_latency = np.random.uniform(min_lat, max_lat)
                latency = base_latency + random.gauss(0, 5)
                latency_data.append({
                    'region1': region1,
                    'region2': region2,
                    'latency_ms': max(latency, 1),
                    'timestamp': timestamp
                })

latency_df = pd.DataFrame(latency_data)
latency_df.to_csv('data/regional_latency.csv', index=False)
print(f"Generated {len(latency_df)} latency measurements")
print(latency_df.head())

# ==================== PART 3: Generate Traffic Patterns ====================
print("\nGenerating Traffic Patterns...")

traffic_data = []
start_date = datetime(2024, 1, 1)

for days in range(90):  # 3 months
    for hour in range(24):
        timestamp = start_date + timedelta(days=days, hours=hour)
        
        # Peak hours are 9-17 (business hours)
        hour_of_day = timestamp.hour
        if 9 <= hour_of_day <= 17:
            traffic_multiplier = random.uniform(1.5, 2.5)
        elif 22 <= hour_of_day or hour_of_day <= 6:
            traffic_multiplier = random.uniform(0.2, 0.5)  # low traffic at night
        else:
            traffic_multiplier = random.uniform(0.8, 1.2)
        
        # Weekend traffic is lower
        if timestamp.weekday() >= 5:  # Saturday = 5, Sunday = 6
            traffic_multiplier *= 0.7
        
        for service_id, service_row in services_df.iterrows():
            base_traffic = service_row['traffic_volume_rps']
            
            for region in regions:
                # Different regions have different traffic volumes
                region_factor = {
                    'us-east-1': 1.0,
                    'us-west-2': 0.8,
                    'eu-west-1': 0.6,
                    'ap-southeast-1': 0.5,
                    'ap-northeast-1': 0.4,
                }[region]
                
                requests = int(base_traffic * traffic_multiplier * region_factor)
                
                traffic_data.append({
                    'service_id': service_id + 1,
                    'region': region,
                    'hour': hour,
                    'requests': requests,
                    'timestamp': timestamp
                })

traffic_df = pd.DataFrame(traffic_data)
traffic_df.to_csv('data/traffic_patterns.csv', index=False)
print(f"Generated {len(traffic_df)} traffic records")
print(traffic_df.head())

# ==================== PART 4: Generate Placement History ====================
print("\nGenerating Service Placement History...")

placement_data = []
start_date = datetime(2024, 1, 1)

for days in range(90):
    timestamp = start_date + timedelta(days=days)
    
    for service_id in range(1, len(service_names) + 1):
        service = services_df[services_df['service_id'] == service_id].iloc[0]
        
        # Latency critical services are usually in fewer regions
        if service['latency_critical']:
            num_regions = random.choice([1, 2])
        else:
            num_regions = random.choice([2, 3, 4])
        
        placement_regions = random.sample(regions, num_regions)
        
        for region in placement_regions:
            placement_data.append({
                'service_id': service_id,
                'region': region,
                'timestamp': timestamp,
                'instances': random.randint(1, 5),
                'avg_latency_ms': random.uniform(5, 100),
                'error_rate': random.uniform(0, 0.05)
            })

placement_df = pd.DataFrame(placement_data)
placement_df.to_csv('data/service_placement.csv', index=False)
print(f"Generated {len(placement_df)} placement records")
print(placement_df.head())

# ==================== Summary ====================
print("\n" + "="*50)
print("ALL DATA GENERATED SUCCESSFULLY!")
print("="*50)
print(f"\nFiles created in 'data/' folder:")
print(f"   • services.csv ({len(services_df)} rows)")
print(f"   • regional_latency.csv ({len(latency_df)} rows)")
print(f"   • traffic_patterns.csv ({len(traffic_df)} rows)")
print(f"   • service_placement.csv ({len(placement_df)} rows)")
print(f"\nTotal records generated: {len(services_df) + len(latency_df) + len(traffic_df) + len(placement_df):,}")
