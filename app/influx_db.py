# # influx_db.py

# from influxdb_client import InfluxDBClient, Point, WritePrecision
# import os
# from datetime import datetime

# # InfluxDB configuration
# INFLUXDB_URL = "http://localhost:8086"
# INFLUXDB_TOKEN = "lvSU-8WG2oV51x0dOQ86oIo8lmWReNSoTlic1yrJAgkBj9wQKUTt6zh_n207DDoEyDb07q_03TjcblQmv7ScNg=="
# INFLUXDB_ORG = "Benhive"
# INFLUXDB_BUCKET = "initial"

# # Create InfluxDB client
# client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG, timeout=60)  # Increase timeout to 60 seconds

# def log_user_login(username: str):
#     """Log user login details to InfluxDB."""
#     try:
#         print(username,"fsfs")
#         write_api = client.write_api()
#         point = (
#             Point("user_logins")
#             .tag("username", username)
#             .field("login_time",datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
#         )
#         write_api.write(bucket=INFLUXDB_BUCKET, record=point)
#         print(f"Logged login for user: {username}")
#     except Exception as e:
#         print(f"Failed to log login for user {username}: {e}")
