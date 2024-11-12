from sqlalchemy import Column, Integer, String, TIMESTAMP, ForeignKey, func,Float,Text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSON
from db import Base

class Roles(Base):
    __tablename__ = 'roles'
    role_id = Column(Integer, primary_key=True, index=True)
    role = Column(String(50), unique=True, nullable=False)  # Role should be unique and non-null
    status = Column(Integer, nullable=False, default=1)  # Assuming 'status' is a flag (e.g., active/inactive)
class User(Base):
    __tablename__ = 'users'

    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)  # Hashed password
    email = Column(String, unique=True, nullable=False)
    role_id = Column(Integer, ForeignKey('roles.role_id'), nullable=False)  # Foreign key relationship to roles table
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())  # Automatically set current timestamp
    updated_at = Column(TIMESTAMP, nullable=False, server_default=func.now(), onupdate=func.now())  # Auto-update on change

    # Relationship to the Roles model
    role = relationship("Roles")

class Property(Base):
    __tablename__ = 'properties'

    property_id = Column(Integer, primary_key=True, index=True)
    state = Column(String, nullable=False)  # Added new field for 'state'
    city = Column(String, nullable=True)  # Optional city field
    property_type = Column(String, nullable=True)  # Optional property_type field
    bedrooms = Column(Integer, nullable=True, default=0)  # Optional bedrooms field with default
    bathrooms = Column(Integer, nullable=True, default=0)  # Optional bathrooms field with default
    description = Column(Text, nullable=True, default="")  # Optional description field
    price = Column(Float, nullable=True, default=0.0)  # Optional price field with default
    price_range = Column(String, nullable=True, default="")  # Optional price range field
    features = Column(JSON, nullable=True)  # JSON to hold list of features
    nearby = Column(JSON, nullable=True)  # JSON to hold list of nearby amenities
    year_built = Column(Integer, nullable=False)  # Original 'year_built' field (mandatory)
    amount = Column(Float, nullable=False)  # Original 'amount' field (mandatory)