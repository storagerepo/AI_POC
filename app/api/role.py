from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

# Import our Role model and schemas
from models import Roles
from db import get_db
from pydantic import BaseModel
router = APIRouter()

class RoleCreate(BaseModel):
    role: str
    status: int = None

class RoleRead(BaseModel):
    id: int
    role: str
    status: int = None

    class Config:
        orm_mode = True

# Create a new role
@router.post("/create_role", response_model=RoleRead)
def create_role(role: RoleCreate, db: Session = Depends(get_db)):
    # Check if role already exists
    db_role = db.query(Roles).filter(Roles.role == role.role).first()
    if db_role:
        raise HTTPException(status_code=400, detail="Role already exists")

    new_role = Roles(role=role.role, status=role.status)
    db.add(new_role)
    db.commit()
    db.refresh(new_role)
    return new_role

# Get all roles
@router.get("/roles", response_model=List[RoleRead])
def get_roles(db: Session = Depends(get_db)):
    roles = db.query(Roles).all()
    return roles

# Get role by ID
@router.get("/roles/{role_id}", response_model=RoleRead)
def get_role(role_id: int, db: Session = Depends(get_db)):
    role = db.query(Roles).filter(Roles.id == role_id).first()
    if role is None:
        raise HTTPException(status_code=404, detail="Role not found")
    return role

# Update a role
@router.put("/roles/{role_id}", response_model=RoleRead)
def update_role(role_id: int, role: RoleCreate, db: Session = Depends(get_db)):
    db_role = db.query(Roles).filter(Roles.id == role_id).first()
    if db_role is None:
        raise HTTPException(status_code=404, detail="Role not found")

    db_role.role = role.role
    db_role.status = role.status
    db.commit()
    db.refresh(db_role)
    return db_role

# Delete a role
@router.delete("/roles/{role_id}", response_model=dict)
def delete_role(role_id: int, db: Session = Depends(get_db)):
    db_role = db.query(Roles).filter(Roles.id == role_id).first()
    if db_role is None:
        raise HTTPException(status_code=404, detail="Role not found")

    db.delete(db_role)
    db.commit()
    return {"message": "Role deleted successfully"}
