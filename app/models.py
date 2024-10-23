from sqlalchemy import Column, Integer, String, TIMESTAMP, ForeignKey, func
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

from db import Base

class Roles(Base):
    __tablename__ = 'roles'
    role_id = Column(Integer, primary_key=True, index=True)
    role = Column(String(50), unique=True, nullable=False)  # Role should be unique and non-null
    status = Column(Integer, nullable=False, default=1)  # Assuming 'status' is a flag (e.g., active/inactive)

    # # Establish a relationship to the User model
    # users = relationship('User', back_populates='role')


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