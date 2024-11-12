import datetime
from fastapi import Depends, HTTPException, status
from fastapi import APIRouter, Depends, HTTPException,Form,Cookie,Response,Request
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer
from models import User  # Adjust import based on your app structure
from db import get_db  # Your function to get the DB session
from jose import jwt, JWTError

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def role_required(required_roles: list):
    async def role_dependency(request: Request,db: Session = Depends(get_db)):
        try:
            
            # Decode the JWT token to get unverified claims
            token = request.cookies.get('access_token')

            unverified_claims = jwt.get_unverified_claims(token)
            user_email = unverified_claims.get("email")
            print(user_email)
            # Fetch the user from the database
            user = db.query(User).filter(User.email == user_email).first()
            if user is None:
                return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail={"status":False,"message":"User not found"})

            user_roles = [user.role.role]  # Assuming user has a roles relationship
            if not any(role in required_roles for role in user_roles):
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail={"status":False,"message":"Insufficient permissions"})

            return user  # Return the user object if role check passes
        except JWTError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail={"status":False,"message":"Invalid token"})
    
    return role_dependency  # Return the inner function


