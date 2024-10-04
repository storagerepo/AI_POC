from fastapi import FastAPI
from api.pricePrediction import router as pricePredictionRouter
from api.propertyRecommendations import router as propertyRecommendation
from api.chatbot import router as chatBotRouter
import uvicorn
app = FastAPI(debug=True)

app.include_router(pricePredictionRouter, prefix="/api/pricePredictions")
app.include_router(propertyRecommendation,prefix="/api/propertyRecommendation")
app.include_router(chatBotRouter,prefix='/api/chatBot')


@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI application!"}

if __name__ == "__main__":
    # Start the FastAPI app using Uvicorn with the desired host and port
    uvicorn.run("main:app", host="127.0.0.1", port=8004, reload=True)