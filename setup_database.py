import sqlite3
import pandas as pd
import os

print("Setting up SQLite Database...\n")

# Create/connect to database
db_path = 'resource_optimization.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print(f"Connected to database: {db_path}\n")

# ==================== Load Services ====================
print("Loading services.csv...")
services_df = pd.read_csv('data/services.csv')
services_df.to_sql('services', conn, if_exists='replace', index=False)
print(f"Loaded {len(services_df)} services\n")

# ==================== Load Regional Latency ====================
print("Loading regional_latency.csv...")
latency_df = pd.read_csv('data/regional_latency.csv')
latency_df['timestamp'] = pd.to_datetime(latency_df['timestamp'])
latency_df.to_sql('regional_latency', conn, if_exists='replace', index=False)
print(f"Loaded {len(latency_df)} latency records\n")

# ==================== Load Traffic Patterns ====================
print("Loading traffic_patterns.csv...")
traffic_df = pd.read_csv('data/traffic_patterns.csv')
traffic_df['timestamp'] = pd.to_datetime(traffic_df['timestamp'])
traffic_df.to_sql('traffic_patterns', conn, if_exists='replace', index=False)
print(f"Loaded {len(traffic_df)} traffic records\n")

# ==================== Load Service Placement ====================
print("Loading service_placement.csv...")
placement_df = pd.read_csv('data/service_placement.csv')
placement_df['timestamp'] = pd.to_datetime(placement_df['timestamp'])
placement_df.to_sql('service_placement', conn, if_exists='replace', index=False)
print(f"Loaded {len(placement_df)} placement records\n")

# ==================== Create Indexes (for faster queries) ====================
print("Creating indexes for faster queries...")
cursor.execute('CREATE INDEX IF NOT EXISTS idx_service_id ON services(service_id)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_service_placement_service ON service_placement(service_id)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_traffic_service ON traffic_patterns(service_id)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_latency_regions ON regional_latency(region1, region2)')
print("Indexes created\n")

conn.commit()

# ==================== Verify Data ====================
print("="*60)
print("DATABASE SETUP COMPLETE!")
print("="*60)

# Show table info
tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print(f"\nTables in database ({len(tables)}):")
for table in tables:
    count = cursor.execute(f"SELECT COUNT(*) FROM {table[0]}").fetchone()[0]
    print(f"   • {table[0]}: {count:,} rows")


conn.close()