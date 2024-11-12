import asyncio
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException,Form,Cookie,Response,Request
from sqlalchemy.orm import Session
from pydantic import BaseModel
from db import get_db
from models import User,Roles
from api.auth import role_required
import httpx
from fastapi.security import OAuth2PasswordBearer
import json
from msal import ConfidentialClientApplication
# from influx_db import log_user_login,client
# from influxdb_client.client.query_api import QueryApi
from keycloak.exceptions import KeycloakPostError,KeycloakGetError
from fastapi.responses import JSONResponse
from keycloak_setup import keycloak_admin,keycloak_openid
from config import settings
import requests
import time
from jose import jwt
from datetime import datetime

router = APIRouter()
# query_api = client.query_api()
# Input and response models
class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    firstName:str
    lastName:str
    role_id:int # Add roles field
class UserOut(BaseModel):
    user_id: int
    username: str
    email: str
    created_at: datetime  
    updated_at: datetime  

    class Config:
        orm_mode = True
class UserLogin(BaseModel):
    username: str
    password: str

class GoogleLogin(BaseModel):
    token: str

class MicrosoftLogin(BaseModel):
    token: str

class Token(BaseModel):
    access_token: str
    token_type: str


def get_tokens(username, password):
    try:
        # Fetch access token and refresh token
        token = keycloak_openid.token(username, password)
        return token
    except Exception as e:
        print(f"Error while logging in: {str(e)}")
        raise

def clear_user_required_actions(user_id):
    try:
        keycloak_admin.update_user(user_id, {
            "requiredActions": []
        })
    except Exception as e:
        print(f"Failed to clear required actions: {str(e)}")
@router.post("/register")
def register(user_create: UserCreate, db: Session = Depends(get_db)):
    try:
        # Attempt to create the user in Keycloak
        user_id = keycloak_admin.create_user({
            "username": user_create.username,
            "email": user_create.email,
            "firstName": user_create.firstName,
            "lastName": user_create.lastName,
            "enabled": True,
            "credentials": [{
                "type": "password",
                "value": user_create.password,
                "temporary": False
            }]
        })

        # Optionally verify the email and clear any required actions for the user
        keycloak_admin.update_user(user_id, {"emailVerified": True})
        clear_user_required_actions(user_id)

    except KeycloakPostError as e:
        # Check if the error indicates a conflict due to existing user
        if e.response_code == 409:
            return JSONResponse(
                status_code=400,
                content={
                    "status": False,
                    "message": "User already exists with the same email",
                    "code": 400
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "status": False,
                    "message": f"Failed to create user in Keycloak: {str(e)}",
                    "code": 500
                }
            )

    # Store user information in your local database
    try:
        user = User(
            username=user_create.username,
            email=user_create.email,
            role_id=user_create.role_id
        )

        # Add and commit the new user to the local database
        db.add(user)
        db.commit()
        db.refresh(user)

    except Exception as e:
        # If database operation fails, rollback the session and raise HTTP exception
        db.rollback()  # Rollback the transaction on error
        return JSONResponse(
            status_code=500,
            content={
                "status": False,
                "message": f"Failed to register user in local database: {str(e)}",
                "code": 500
            }
        )

    return {
        "status": True,
        "message": "User registered successfully",
        "user": {
            "username": user.username,
            "email": user.email
        }
    }

@router.post("/login")
def login(
    username: str = Form(...),  # Get username from form data
    password: str = Form(...),  # Get password from form data
    db: Session = Depends(get_db),
    remember_me: bool = Form(False),  # Include remember_me checkbox
    response: Response = None
):
    # Generate the JWT access token
    token = get_tokens(username, password)

    unverified_claims = jwt.get_unverified_claims(token['access_token'])
   # Access token and refresh token from the token response
    access_token = token['access_token']
    refresh_token = token['refresh_token']
    user_details = {"username": username, "email": unverified_claims['email'],"access_token": access_token,
        "refresh_token": refresh_token}
    response = JSONResponse(content=user_details, status_code=200)    
    refresh_expires = 3600 * 24 * 30 if remember_me else 3600  # 30 days vs. 1 hour
    response.set_cookie('refresh_token', refresh_token, httponly=True, expires=refresh_expires, path='/', samesite='None', secure=True)
    response.set_cookie('access_token', access_token, httponly=True, expires=60, path='/', samesite='None', secure=True)
    return response

@router.post("/refresh")
def refresh_token(
    response: Response,
    refresh_token: str = Cookie(None)
):
    
    if refresh_token is None:
        raise HTTPException(status_code=401, detail="Refresh token not provided")

    try:
        # Fetch a new access token using the refresh token
        new_tokens = keycloak_openid.refresh_token(refresh_token)
        # Set new tokens as cookies
        response.set_cookie(key="access_token", value=new_tokens['access_token'], httponly=True, expires=60)
        response.set_cookie(key="refresh_token", value=new_tokens['refresh_token'], httponly=True, expires=3600)

        return {"access_token": new_tokens['access_token'], "token_type": "bearer"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh token: {str(e)}")


def get_token_expiry(token: str) -> int:
    # Decode the token to get the expiration time
    payload = jwt.decode(token, keycloak_openid.public_key, algorithms=["RS256"])  # Use the public key of your Keycloak
    return payload.get("exp")

# Function to automatically refresh access tokens in the background
async def auto_refresh_token(request: Request,refresh_token: str, response: Response):
    while True:
        await asyncio.sleep(30)  # Check every 30 seconds
        
        # Fetch the current access token from cookies
        access_token = request.cookies.get('access_token')

        if access_token:
            expiry_time = get_token_expiry(access_token)
            time_left = expiry_time - time.time()

            if time_left < 60:  # Refresh token 1 minute before it expires
                new_tokens = await refresh_token(response, access_token=access_token, refresh_token=refresh_token)
                access_token = new_tokens['access_token']  # Update the token for the next iteration

@router.get("/getAllUsers", response_model=List[UserOut])
async def get_all_users(
    db: Session = Depends(get_db), 
    user: User = Depends(role_required(["PORTAL_USER"]))  # Await the role_required function directly
):
    # Query all users from the database
    users = db.query(User).all()  # Fetch all user records
    return users

@router.post("/logout")
async def logout(request: Request, response: Response = None):
    # Clear the cookies for access_token and refresh_token
   

    # Get the refresh token from the request cookies
    refresh_token = request.cookies.get('refresh_token')

    # Optionally invalidate the session on Keycloak
    keycloak_logout_url = "http://localhost:8080/realms/OBR/protocol/openid-connect/logout"
    
    logout_data = {
        'client_id': settings.keycloak_client_id,
        'refresh_token': refresh_token,
        'client_secret': settings.keycloak_client_secret,  # Ensure you have the correct secret
    }
    
    try:
        # Make a request to Keycloak's logout endpoint
        async with httpx.AsyncClient() as client:
            await client.post(keycloak_logout_url, data=logout_data)
    except Exception as e:
        print(f"Failed to logout from Keycloak: {e}")
    details={"message": "Logged out successfully"}
    response = JSONResponse(content=details, status_code=200)    

    response.delete_cookie('refresh_token', path='/')
    response.delete_cookie('access_token', path='/')
    return response


async def get_user_info(access_token: str):
    """Fetch user info from Google using the access token."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            response.raise_for_status()  # Raise an error for bad responses
            return response.json()
    
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            raise ValueError("Access token is invalid or expired.")
        else:
            raise ValueError(f"Failed to fetch user info: {e.response.status_code}")
    
    except httpx.RequestError as e:
        raise ValueError(f"An error occurred while requesting user info: {e}")

@router.post("/login/google")
async def login_google(google_login: GoogleLogin, db: Session = Depends(get_db)):
    # Fetch user info using the access token
    user_info = await get_user_info(google_login.token)
    
    # Extract the username and password (split email and use before @ as password)
    username = user_info['email']
    password = (user_info['email']).split('@')[0]
    try:
        # Attempt to create the user in Keycloak
        user_id = keycloak_admin.create_user({
            "username": username,
            "email": user_info['email'],
            "firstName": user_info['given_name'],
            "lastName": user_info['family_name'],
            "enabled": True,
            "credentials": [{
                "type": "password",
                "value": password,
                "temporary": False
            }]
        })

        # Verify email and clear any required actions
        keycloak_admin.update_user(user_id, {"emailVerified": True})
        clear_user_required_actions(user_id)

    except KeycloakPostError as e:
        # If user already exists in Keycloak (409 conflict error)
        if e.response_code == 409:
            # Authenticate existing user to get an access token
            token = get_tokens(username, password)
            user_details = {"username": username, "email":user_info['email'],"access_token": token['access_token'],
            "refresh_token": token['refresh_token']}
            response = JSONResponse(content=user_details, status_code=200)    
            access_token = token['access_token']
            refresh_token = token['refresh_token']
            refresh_expires = 3600  # 30 days vs. 1 hour
            response.set_cookie('refresh_token', refresh_token, httponly=True, expires=refresh_expires, path='/', samesite='None', secure=True)
            response.set_cookie('access_token', access_token, httponly=True, expires=60, path='/', samesite='None', secure=True)
            return response
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "status": False,
                    "message": f"Failed to create or login user in Keycloak: {str(e)}",
                    "code": 500
                }
            )

    # If new user registered, authenticate and return access token
    token = keycloak_openid.token(username=username, password=password)
    user_details = {"username": username, "email":user_info['email'],"access_token": token['access_token'],
            "refresh_token": token['refresh_token']}
    response = JSONResponse(content=user_details, status_code=200)    
    access_token = token['access_token']
    refresh_token = token['refresh_token']
    refresh_expires = 3600  # 30 days vs. 1 hour
    response.set_cookie('refresh_token', refresh_token, httponly=True, expires=refresh_expires, path='/', samesite='None', secure=True)
    response.set_cookie('access_token', access_token, httponly=True, expires=60, path='/', samesite='None', secure=True)
    # Optionally, save user in local DB for future references
# Query the role
    role = db.query(Roles).filter(Roles.role == 'PORTAL_USER').first()

# Check if the role was found
    if role is None:
        print("Role 'PORTAL_USER' not found in the database.")
        # Optionally, handle this case, e.g., by raising an exception or returning early
        raise ValueError("Role 'PORTAL_USER' does not exist")

    # Ensure user_info contains the necessary data
    if 'username' not in user_details or 'email' not in user_details:
        raise ValueError("user_info must contain 'username' and 'email'")
    print(user_details,role)
    # Create and add the new user
    user = User(
        username=user_details['username'],
        email=user_details['email'],
        role_id=role.role_id
    )

    # Add and commit the user to the database
    try:
        db.add(user)
        db.commit()
        db.refresh(user)
        print("User successfully added:", user)
    except Exception as e:
        db.rollback()  # Rollback the transaction in case of an error
        print("Error occurred while adding the user:", str(e))


    return response
