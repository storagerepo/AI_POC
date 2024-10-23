from fastapi import FastAPI, Depends
from db import database, get_db, create_tables,check_connection  # Import create_tables function
from api.pricePrediction import router as pricePredictionRouter
from api.propertyRecommendations import router as propertyRecommendationRouter
from api.chatbot import router as chatBotRouter
from api.role import router as roleRouter# from api.chatbot import router as chatBotRouter
from fastapi.middleware.cors import CORSMiddleware
from api.user import router as userRouter
import uvicorn

app = FastAPI()


# Include routers
app.include_router(pricePredictionRouter, prefix="/api/pricePredictions")
app.include_router(propertyRecommendationRouter, prefix="/api/propertyRecommendations")
app.include_router(chatBotRouter, prefix='/api/chatBot')
app.include_router(userRouter, prefix='/api/user')
app.include_router(roleRouter, prefix='/api/role')


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to your frontend origin for security
    allow_credentials=True,  # Important to allow cookies
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    print("Starting FastAPI application...")
    await database.connect()
    check_connection() 
    # check_influx_connection() # Connect to the async database
    create_tables()  # Create tables

@app.on_event("shutdown")
async def shutdown():
    print("Shutting down FastAPI application...")
    await database.disconnect()  # Disconnect from the async database


@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI application!"}

if __name__ == "__main__":
    # Start the FastAPI app using Uvicorn with the desired host and port
    uvicorn.run("main:app", host="localhost", port=8000,log_level="debug")
    #Swagger Doc: "http://localhost:8004/docs"

