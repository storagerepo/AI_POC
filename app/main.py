from fastapi import FastAPI
from api.pricePrediction import router as pricePredictionRouter
from api.propertyRecommendations import router as propertyRecommendation
from api.chatbot import router as chatBotRouter
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
app = FastAPI(debug=True)

app.include_router(pricePredictionRouter, prefix="/api/pricePredictions")
app.include_router(propertyRecommendation,prefix="/api/propertyRecommendation")
app.include_router(chatBotRouter,prefix='/api/chatBot')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)


@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI application!"}

if __name__ == "__main__":
    # Start the FastAPI app using Uvicorn with the desired host and port
    uvicorn.run("main:app", host="localhost", port=8004,log_level="debug")
    #Swagger Doc: "http://localhost:8004/docs"