from fastapi import FastAPI
from api.pricePrediction import router as api_router
import uvicorn
app = FastAPI(debug=True)

app.include_router(api_router, prefix="/api/pricePredictions")


@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI application!"}

if __name__ == "__main__":
    # Start the FastAPI app using Uvicorn with the desired host and port
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)