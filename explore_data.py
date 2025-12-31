import sqlite3
import pandas as pd

print("EXPLORING RESOURCE OPTIMIZATION DATA\n")

# Connect to database
conn = sqlite3.connect('resource_optimization.db')

# ==================== QUERY 1: Service Overview ====================
print("="*100)
print("SERVICE OVERVIEW")
print("="*100)

query1 = """
SELECT 
    service_id,
    service_name,
    memory_mb,
    cpu_cores,
    latency_critical,
    traffic_volume_rps,
    dependencies
FROM services
ORDER BY traffic_volume_rps DESC
LIMIT 10
"""

df1 = pd.read_sql_query(query1, conn)
print(df1.to_string(index=False))
print()

# ==================== QUERY 2: Regional Latency Summary ====================
print("="*100)
print("REGIONAL LATENCY MATRIX (average ms)")
print("="*100)

query2 = """
SELECT 
    region1,
    region2,
    ROUND(AVG(latency_ms), 2) as avg_latency_ms,
    ROUND(MIN(latency_ms), 2) as min_latency_ms,
    ROUND(MAX(latency_ms), 2) as max_latency_ms,
    COUNT(*) as samples
FROM regional_latency
GROUP BY region1, region2
ORDER BY region1, region2
"""

df2 = pd.read_sql_query(query2, conn)
print(df2.to_string(index=False))
print()

# ==================== QUERY 3: Traffic by Region ====================
print("="*100)
print("TOTAL TRAFFIC BY REGION")
print("="*100)

query3 = """
SELECT 
    region,
    SUM(requests) as total_requests,
    ROUND(AVG(requests), 0) as avg_hourly_requests,
    COUNT(DISTINCT service_id) as num_services
FROM traffic_patterns
GROUP BY region
ORDER BY total_requests DESC
"""

df3 = pd.read_sql_query(query3, conn)
print(df3.to_string(index=False))
print()

# ==================== QUERY 4: Services by Placement Count ====================
print("="*100)
print("SERVICE PLACEMENT DISTRIBUTION")
print("="*100)

query4 = """
SELECT 
    s.service_id,
    s.service_name,
    COUNT(DISTINCT sp.region) as num_regions,
    ROUND(AVG(sp.avg_latency_ms), 2) as avg_latency_ms,
    ROUND(AVG(sp.error_rate), 4) as avg_error_rate
FROM services s
LEFT JOIN service_placement sp ON s.service_id = sp.service_id
GROUP BY s.service_id
ORDER BY num_regions DESC, s.service_name
"""

df4 = pd.read_sql_query(query4, conn)
print(df4.to_string(index=False))
print()

# ==================== QUERY 5: Peak Traffic Hours ====================
print("="*100)
print("PEAK TRAFFIC HOURS (all regions combined)")
print("="*100)

query5 = """
SELECT 
    hour,
    SUM(requests) as total_requests,
    ROUND(AVG(requests), 0) as avg_requests_per_service_region
FROM traffic_patterns
GROUP BY hour
ORDER BY total_requests DESC
LIMIT 10
"""

df5 = pd.read_sql_query(query5, conn)
print(df5.to_string(index=False))
print()

# ==================== QUERY 6: Cross-Region Traffic Analysis ====================
print("="*100)
print("HIGH LATENCY REGION PAIRS (average > 100ms)")
print("="*100)

query6 = """
SELECT 
    region1,
    region2,
    ROUND(AVG(latency_ms), 2) as avg_latency_ms
FROM regional_latency
GROUP BY region1, region2
HAVING AVG(latency_ms) > 100
ORDER BY avg_latency_ms DESC
"""

df6 = pd.read_sql_query(query6, conn)
print(df6.to_string(index=False))
print()

# ==================== QUERY 7: Latency Critical Services ====================
print("="*100)
print("LATENCY CRITICAL SERVICES")
print("="*100)

query7 = """
SELECT 
    service_id,
    service_name,
    memory_mb,
    traffic_volume_rps,
    dependencies
FROM services
WHERE latency_critical = 1
ORDER BY traffic_volume_rps DESC
"""

df7 = pd.read_sql_query(query7, conn)
print(df7.to_string(index=False))
print()

# ==================== SUMMARY STATS ====================
print("="*100)
print("SUMMARY STATISTICS")
print("="*100)

query_summary = "SELECT COUNT(*) as total_services FROM services"
total_services = pd.read_sql_query(query_summary, conn).iloc[0, 0]

query_summary = "SELECT COUNT(DISTINCT region) as num_regions FROM traffic_patterns"
num_regions = pd.read_sql_query(query_summary, conn).iloc[0, 0]

query_summary = "SELECT SUM(requests) as total_traffic FROM traffic_patterns"
total_traffic = pd.read_sql_query(query_summary, conn).iloc[0, 0]

query_summary = "SELECT ROUND(AVG(latency_ms), 2) as avg_latency FROM regional_latency"
avg_latency = pd.read_sql_query(query_summary, conn).iloc[0, 0]

print(f"• Total Services: {total_services}")
print(f"• Total Regions: {num_regions}")
print(f"• Total Traffic Records: {total_traffic:,}")
print(f"• Average Cross-Region Latency: {avg_latency} ms")
print()

conn.close()

print("="*100)
print("✅ DATA EXPLORATION COMPLETE!")
print("="*100)
