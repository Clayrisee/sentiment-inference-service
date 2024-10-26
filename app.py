from fastapi import FastAPI
from joblib import load
from model import InferenceModel 
from datetime import datetime
import os

app = FastAPI()

model = InferenceModel()

@app.get("/")
async def predict(text: str):
    result_predict = model.predict(text=text)
    # Get the current timestamp and format it
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Add timestamp
    result_predict["timestamp"] = timestamp 
    return result_predict

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)