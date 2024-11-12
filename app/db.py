# db.py

from sqlalchemy import create_engine,text
from sqlalchemy.orm import sessionmaker, declarative_base
from databases import Database
import os
import influxdb_client
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/Demo")

# SQLAlchemy specific setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Models Base
Base = declarative_base()

# Async database connection
database = Database(DATABASE_URL)
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(bind=engine)
# InfluxDB configuration
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "lvSU-8WG2oV51x0dOQ86oIo8lmWReNSoTlic1yrJAgkBj9wQKUTt6zh_n207DDoEyDb07q_03TjcblQmv7ScNg==" # Ensure you set this in your environment
INFLUXDB_ORG = "Benhive"
INFLUXDB_BUCKET = "initial"

# InfluxDB client setup
influxdb_client_instance = influxdb_client.InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)

# Dependency to get the DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    # Create all tables in the database
   print( Base.metadata.create_all(bind=engine))
def check_connection():
   try:
        with engine.connect() as connection:
            # Use the text() function to create an executable SQL expression
            connection.execute(text("SELECT 1"))  # Execute a simple query
            print("Database connection successful!")
   except Exception as e:
        print(f"Database connection failed: {e}")

def check_influx_connection():
    try:
        # Example of writing a point to verify the connection
        write_api = influxdb_client_instance.write_api()
        point = influxdb_client.Point("test_measurement").field("test_field", 123)
        write_api.write(INFLUXDB_BUCKET, record=point)
        print("InfluxDB connection successful!")
    except Exception as e:
        print(f"InfluxDB connection failed: {e}")